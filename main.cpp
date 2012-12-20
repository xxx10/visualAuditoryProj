#include "opencv/cv.h"
#include "opencv/highgui.h"
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
using namespace std;
static CvMemStorage* storage = 0;		
static CvHaarClassifierCascade* cascade = 0;		//������
typedef struct __faceinfo {
  double x;
  double y;
  double width;
  double height;
  bool isSpeaking;
}FaceInfo;
static FaceInfo result[50000][10];
static int frameNum = 0;
static FaceInfo refer[5];
static int people = 2;

int is_speaking(int people ,FaceInfo *face); 
void detect_and_draw( IplImage* image ); 
void beautify_result(FaceInfo result[50000][10]);
void sort_horizontally(FaceInfo face[50000][10]);
void swap_faceinfo(FaceInfo face1, FaceInfo face2);
void init_params();
bool considerable(FaceInfo face);
const char* cascade_name = "haarcascade_frontalface_alt.xml";		//������·��
/*    "haarcascade_profileface.xml";*/
 
int main( int argc, char** argv )
{
  init_params();
  CvCapture* capture = 0;
  IplImage *frame, *frame_copy = 0;
  int optlen = strlen("--cascade="); //?  --cascade=Ϊ������ѡ��ָʾ���� ?
  const char* input_name;
    
  cascade_name = "haarcascade_frontalface_alt2.xml";   //load classifierCascade
  input_name = argc > 1 ? argv[1] : 0;
  cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
  if( !cascade )		//���ط���������
    {
      fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
      return -1;
    }
  storage = cvCreateMemStorage(0);		// �����ڴ�洢�����ڴ��
 
  if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') )		//������ͷ��ȡ
    //      capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );	//��ʼ������ͷ��׽��
    capture = cvCaptureFromAVI("Vid1.avi");

  cvNamedWindow( "result", 1 );  //��������
 
  if( capture )    //����Ƶ�ļ�(����ͷ)��֡����
    {
      for(frameNum = 0;++frameNum;)
        {
	  if( !cvQueryFrame( capture ))		
	    break;
	  frame = cvRetrieveFrame( capture ); 
	  if( !frame )		                //ץ֡ʧ��
	    break;
	  if( !frame_copy )		//frame_copyΪ0�����ʼ����֡(Image)�ĸ���
	    frame_copy = cvCreateImage( cvSize(frame->width,frame->height),
					IPL_DEPTH_8U, frame->nChannels );
	  if( frame->origin == IPL_ORIGIN_TL )//����ԭ��Ϊ���Ͻ�
	    cvCopy( frame, frame_copy, 0 );
	  else								//������ת����
	    cvFlip( frame, frame_copy, 0 );
	  detect_and_draw( frame_copy );		// ���ü��ͻ��ƺ���
	  is_speaking( people, result[frameNum] );
	  if( cvWaitKey( 10 ) >= 0 )          //�ȴ�����10ms
	    break;
        }
      cvReleaseImage( &frame_copy );			//�ͷ�֡�����ڴ�ռ�
      cvReleaseCapture( &capture );			//�ͷŲ�׽����ͬʱ�ͷ�frame
    }
  cvDestroyWindow("result");		//���ٴ���
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
				       8, 1 );		//cvRound(double i); ��double�������������룬��������
  int i;
 
  cvCvtColor( img, gray, CV_BGR2GRAY );		//��rgbͼ��ת��Ϊ�Ҷ�ͼ�� img-->gray
  cvResize( gray, small_img, CV_INTER_LINEAR );		//Ϊ������������ͼ��ͬ����С����Ҫ��ͼ��������� gray-->small_img
  cvEqualizeHist( small_img, small_img );		//ֱ��ͼ���⻯
  cvClearMemStorage( storage );		//����ڴ��
 
  if( cascade )
    {
      double t = (double)cvGetTickCount();		//����tics����
      CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,				//!���ͼ���е�Ŀ��!
					  1.1, 2, //0,
					  CV_HAAR_DO_CANNY_PRUNING,   
					  //(Ҳ����Ϊ��ֵ0)������ʽ����������Canny��Ե��������ų�һЩ��Ե���ٻ��ߺܶ��ͼ��������Ϊ����������һ�㲻������Ŀ��
					  cvSize(80, 80) );
      // t = (double)cvGetTickCount() - t;		//ͳ�Ƽ��ʱ��
      //      printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
      for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
	  CvRect* r = (CvRect*)cvGetSeqElem( faces, i );		//��faces���ݴ�CvSeqתΪCvRect ?
	  //just for human eyes
	  CvPoint center;
	  int radius;
	  center.x = cvRound(r->x + r->width*0.5);
	  center.y = cvRound(r->y + r->height*0.5);
	  radius = cvRound((r->width + r->height)*0.25);
	  cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
	  
	  FaceInfo tmp;
	  tmp.x = r->x;
	  tmp.y = r->y;
	  tmp.width = r->width;
	  tmp.height = r->height;
	  if(considerable(tmp))
	    cout<<"considerable"<<endl;
	  else break;
	  result[frameNum][i].x = r->x;
	  result[frameNum][i].y = r->y;
	  result[frameNum][i].width = r->width;
	  result[frameNum][i].height = r->height;

	  printf("%f, %f, %f, %f\t", result[frameNum][i].x, result[frameNum][i].y, result[frameNum][i].width, result[frameNum][i].height);
        }
      //t = (double)cvGetTickCount() - t;		//ͳ�Ƽ�⼰����ʱ��
      //      printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
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

void beautify_result(FaceInfo result[50000][10])
{
  sort_horizontally(result);
}

void sort_horizontally(FaceInfo result[50000][10])
{
  for(int j = 0; j < frameNum; j++)
    for(int i = 0;i < 10; i++)
      {
	if(result[j][i].x < result[j][i+1].x)
	  {
	    swap_faceinfo(result[j][i], result[j][i+1]);
	  }
      }
}

void swap_faceinfo(FaceInfo face1, FaceInfo face2)
{
  FaceInfo tmp;
  if(face1.x > face2.x)
    {
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
}

void init_params()
{
  refer[0].x = 197.0;
  refer[0].y = 173.0;
  refer[1].x = 982.0;
  refer[1].y = 242.0;
}

bool considerable(FaceInfo face)
{
  double distance(FaceInfo face1, FaceInfo face2);
  for(int i = 0; i<people; i++)
    {
      if(distance(refer[i], face) < 200)
	return true;
    }
  return false;
}

double distance(FaceInfo face1, FaceInfo face2)
{
  return sqrt((face1.x - face2.x)*(face1.x - face2.x)+(face1.y - face2.y)*(face1.y - face2.y));
}
