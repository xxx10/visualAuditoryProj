#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cmath>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#ifdef _EiC
#define WIN32
#endif
#define CLEAR_FACEINFO(f) f.x=0;f.y=0;f.width=0;f.height=0;f.isSpeaking=0;
#define FRAME_NUM_MAX 10000
using namespace std;
static CvMemStorage* storage = 0;		
static CvHaarClassifierCascade* cascade = 0;		//分类器
typedef struct __faceinfo {
  double x;
  double y;
  double width;
  double height;
  int isSpeaking;
}FaceInfo;
typedef struct _point3{
	double x;
	double y;
	double z;
}point3;
static FaceInfo result[FRAME_NUM_MAX][10];
static int totalFrame = 0;
static int dataLength = 0;//视听数据长度，单位s
static int frameNum = 0;
static FaceInfo refer[5];
static int people = 2;
static point3 micLoc[4];
static char validFile[128];
static char videoFile[128];
static char audioFile[4][128];
static point3 speakerLoc[5]; //最多5人
int is_speaking(int people ,FaceInfo *face); 
void detect_and_draw( IplImage* image ); 
void beautify_result(FaceInfo result[FRAME_NUM_MAX][10]);
void sort_horizontally(FaceInfo face[FRAME_NUM_MAX][10]);
void swap_faceinfo(FaceInfo &face1, FaceInfo &face2);
void init_params(char* filename);
bool considerable(FaceInfo &face);
double distance(FaceInfo face1, FaceInfo face2);
const char* cascade_name = "haarcascade_frontalface_alt.xml";		//分类器路径
//    "haarcascade_profileface.xml";


int main(int argc, char** argv )
{
  init_params(argv[1]);
  CvCapture* capture = 0;
  IplImage *frame, *frame_copy = 0;
  int optlen = strlen("--cascade="); //?  --cascade=为分类器选项指示符号 ?
  const char* input_name;
    
  cascade_name = "haarcascade_frontalface_alt2.xml";   //load classifierCascade
  input_name = argc > 1 ? argv[1] : 0;
  cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
  if( !cascade )		//加载分类器出错
    {
      fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
      return -1;
    }
  storage = cvCreateMemStorage(0);		// 创建内存存储器，内存块
 
//  下面这一步须注释掉xxx12.22
//  if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') )		//从摄像头读取
    //      capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );	//初始化摄像头捕捉器
    capture = cvCaptureFromAVI(videoFile);

  cvNamedWindow( "result", 1 );  //创建窗口
 
  if( capture )    //对视频文件(摄像头)逐帧处理
    {
      for(frameNum = 0;;frameNum++)
        {
	  if( !cvQueryFrame( capture ))		
	    break;
	  frame = cvRetrieveFrame( capture ); 
	  if( !frame )		                //抓帧失败
	    break;
	  if( !frame_copy )		//frame_copy为0，则初始化此帧(Image)的副本
	    frame_copy = cvCreateImage( cvSize(frame->width,frame->height),
					IPL_DEPTH_8U, frame->nChannels );
	  if( frame->origin == IPL_ORIGIN_TL )//像素原点为左上角
	    cvCopy( frame, frame_copy, 0 );
	  else								//右上则反转拷贝
	    cvFlip( frame, frame_copy, 0 );
	  detect_and_draw( frame_copy );		// 调用检测和绘制函数
	  is_speaking( people, result[frameNum] );
	  if( cvWaitKey( 10 ) >= 0 )          //等待键盘10ms
	    break;
        }
      cvReleaseImage( &frame_copy );			//释放帧副本内存空间
      cvReleaseCapture( &capture );			//释放捕捉器，同时释放frame
    }
  cvDestroyWindow("result");		//销毁窗口
  beautify_result(result);
  return 0;
}

void detect_and_draw( IplImage* img )
{
  static CvScalar colors[] = 
    {
      {{0,0,255}},
      {{0,128,255}},
      {{0,255,255}},
      {{0,255,0}},
      {{255,128,0}},
      {{255,255,0}},
      {{255,0,0}},
      {{255,0,255}}
    };
 
  IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
  IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width),
					       cvRound (img->height)),
				       8, 1 );		//cvRound(double i); 对double型数据四舍五入，返回整形
  int i;
 
  cvCvtColor( img, gray, CV_BGR2GRAY );		//将rgb图像转化为灰度图像 img-->gray
  cvResize( gray, small_img, CV_INTER_LINEAR );		//为了让所有输入图像同样大小，需要对图像进行缩放 gray-->small_img
  cvEqualizeHist( small_img, small_img );		//直方图均衡化
  cvClearMemStorage( storage );		//清空内存块
 
  if( cascade )
    {
      double t = (double)cvGetTickCount();		//返回tics个数
      CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,				//!检测图像中的目标!
					  1.1, 2, //0,
					  CV_HAAR_DO_CANNY_PRUNING,   
					  //(也可设为阈值0)操作方式，函数利用Canny边缘检测器来排除一些边缘很少或者很多的图像区域，因为这样的区域一般不含被检目标
					  cvSize(80, 80) );
      // t = (double)cvGetTickCount() - t;		//统计监测时间
      //      printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
      for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
	  CvRect* r = (CvRect*)cvGetSeqElem( faces, i );		//将faces数据从CvSeq转为CvRect ?
	  //just for human eyes
	  CvPoint center;
	  int radius;
	  center.x = cvRound(r->x + r->width*0.5);
	  center.y = cvRound(r->y + r->height*0.5);
	  radius = cvRound((r->width + r->height)*0.25);
	  cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
	  
	  result[frameNum][i].x = r->x;
	  result[frameNum][i].y = r->y;
	  result[frameNum][i].width = r->width;
	  result[frameNum][i].height = r->height;

        }
      printf("%d frame", frameNum);
      printf("\n");
    }

  cvShowImage( "result", img );
  cvReleaseImage( &gray );
  cvReleaseImage( &small_img );
}

int is_speaking(int people ,FaceInfo *face)
{
  face[0].isSpeaking = 1;
  face[1].isSpeaking = 0;
  return 0;
}

void beautify_result(FaceInfo result[FRAME_NUM_MAX][10])
{
  double distance(FaceInfo face1, FaceInfo face2);
  void getRef(FaceInfo result[FRAME_NUM_MAX][10]);
  sort_horizontally(result);
  getRef(result);
  for(int i = 0 ; i<frameNum ; i++)
    for(int j = 0 ; j<10 ; j++)
      {
	if(!considerable(result[i][j]))
	  {
	    result[i][j].x = 2000;
	    result[i][j].y = 2000;
	  }
      }
  sort_horizontally(result);
  for(int j = 0; j<people; j++)
    for(int i = 1; i<frameNum - 1; i++)
      {
	if(distance(result[i-1][j], result[i+1][j]) < 10 && distance(result[i][j], result[i+1][j]) > 30 && result[i-1][j].x != 0)
	  {
	    result[i][j].x = result[i+1][j].x;
	    result[i][j].y = result[i+1][j].y;
	    result[i][j].width = result[i+1][j].width;
	    result[i][j].height = result[i+1][j].width;
	  }
      }
  FILE *valid;
  int c;
  valid = fopen(validFile,"r");
  for(int i = 0; i < frameNum; i++)
    {
      fscanf(valid, "%d", &c);
      if(c == 0)
	{
	  for(int j = 0; j<people; j++)
	    {
	      CLEAR_FACEINFO(result[i][j]);
	    }
	}
    }
  FILE *result_output;
  result_output = fopen("result.dat", "w");
  for(int i = 0; i < frameNum ; i++)
    {
      for(int j = 0 ; j < people ; j++)
      {
	fprintf(result_output, "%f\t%f\t%f\t%f\t%d\t", result[i][j].x, result[i][j].y, result[i][j].width, result[i][j].height, result[i][j].isSpeaking);
      }
      fprintf(result_output, "\n");
    }
}


void sort_horizontally(FaceInfo result[FRAME_NUM_MAX][10])
{
  for(int j = 0; j < frameNum; j++)
    for(int i = 0;i < 10; i++)
      for(int p = 0; p < i; p++)
      {
	if(result[j][p].x > result[j][i].x && result[j][i].x != 0)
	  {
	    printf("before swapping:%f, %f\n",result[j][p].x, result[j][i].x);
	    swap_faceinfo(result[j][p], result[j][i]);
	    printf("after  swapping:%f, %f\n", result[j][p].x, result[j][i].x);
	  }
      }
}

void swap_faceinfo(FaceInfo &face1, FaceInfo &face2)
{
  FaceInfo tmp;
  tmp.x = face1.x;
  tmp.y = face1.y;
  tmp.width = face1.width;
  tmp.height = face1.height;
  face1.x = face2.x;
  face1.y = face2.y;
  face1.width = face2.width;
  face1.height = face2.height;
  face2.x = tmp.x;
  face2.y = tmp.y;
  face2.width = tmp.width;
  face2.height = tmp.height;
}

void init_params(char* filename)
{
  refer[0].x = 197.0;
  refer[0].y = 173.0;
  refer[1].x = 982.0;
  refer[1].y = 242.0;
  FILE* fileparam = fopen(filename, "r");
  if (filename == NULL)
  {
	  printf("cannot open file : %s", filename);
	  exit(0);
  }
  int num_video, num_audio;
  int fps, fs;
  point3 camLoc;
  fscanf(fileparam, "%d%d%d", &num_video, &num_audio, &people);
  fscanf(fileparam, "%d", &dataLength);
  fscanf(fileparam, "%d%d", &fps, &fs);
  totalFrame = fps * dataLength;
  fscanf(fileparam, "%s", validFile);
  fscanf(fileparam, "%s", videoFile);
  fscanf(fileparam, "%lf%lf%lf", &camLoc.x, &camLoc.y, &camLoc.z);
  for (int i = 0; i <= 3; i++)
  {
	  fscanf(fileparam, "%s", audioFile[i]);
	  fscanf(fileparam, "%lf%lf%lf", &micLoc[i].x, &micLoc[i].y, &micLoc[i].z);
  }
  for (int i = 0; i <= people - 1; i++)
  {
	  fscanf(fileparam, "%lf%lf%lf", &speakerLoc[i].x, &speakerLoc[i].y, &speakerLoc[i].z);
  }
  fclose(fileparam);
}

bool considerable(FaceInfo &face)
{
  double distance(FaceInfo face1, FaceInfo face2);
  for(int i = 0; i<people; i++)
    {
      if(distance(refer[i], face) < 160)
	return true;
    }
  return false;
}

double distance(FaceInfo face1, FaceInfo face2)
{
  return sqrt((face1.x - face2.x)*(face1.x - face2.x)+(face1.y - face2.y)*(face1.y - face2.y));
}
void getRef(FaceInfo result[FRAME_NUM_MAX][10])
{
  int count = 0;
  double sum_x[5] = {0};
  double sum_y[5] = {0};
  for(int i = 0 ;i < frameNum ; i++)
    {
      for(int j = 0 ; j < people ; j++)
	{
	  if(result[i][j].x == 0)
	    break;
	  sum_x[j] = sum_x[j] + result[i][j].x;
	  sum_y[j] = sum_y[j] + result[i][j].y;
	  count++;
	}
    }
  for(int i = 0; i< 5 ; i++)
    {
      refer[i].x = sum_x[i]/count*people;
      refer[i].y = sum_y[i]/count*people;
    }
  printf("refer point: (%f, %f), (%f, %f)\n", refer[0].x, refer[0].y, refer[1].x, refer[1].y);
}
