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
#include <algorithm>
#ifdef _EiC
#define WIN32
#endif
#define CLEAR_FACEINFO(f) f.x=0;f.y=0;f.width=0;f.height=0;f.isSpeaking=0;
#define FRAME_NUM_MAX 30000
#define V_SOUND 340
void writeToFile();

using namespace std;
static CvMemStorage* storage = 0;		
static CvHaarClassifierCascade* cascade = 0;		//������
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
int pLeftThan(const void* p1, const void* p2)
{
	return  (*(point3*)p1).x - (*(point3*)p2).x;
}
double distanceP3(const point3 &p1, const point3 &p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + \
				(p1.y - p2.y) * (p1.y - p2.y) + \
				(p1.z - p2.z) * (p1.z - p2.z) );
}
static FaceInfo result[FRAME_NUM_MAX][10];
static int totalFrame = 0;
static int dataLength = 0;//�������ݳ��ȣ���λs
static int frameNum = 0;
static FaceInfo refer[5];
static int people = 2;
static int fps = 0;//֡��
static int fs = 0;//����������
static point3 micLoc[4];
static char validFile[128];
static char videoFile[128];
static char audioFile[4][128];
static point3 speakerLoc[5]; //���5��
static int audioDataNum = 0;//ÿ���ļ����������ݸ���
int is_speaking(int people ,FaceInfo result[FRAME_NUM_MAX][10]); 
void detect_and_draw( IplImage* image ); 
void beautify_result(FaceInfo result[FRAME_NUM_MAX][10]);
void sort_horizontally(FaceInfo face[FRAME_NUM_MAX][10]);
void swap_faceinfo(FaceInfo &face1, FaceInfo &face2);
void init_params(char* filename);
void writeToFile();
bool considerable(FaceInfo &face);
double distance(FaceInfo face1, FaceInfo face2);
const char* cascade_name = "haarcascade_frontalface_alt.xml";		//������·��
//    "haarcascade_profileface.xml";


int main(int argc, char** argv )
{
  init_params(argv[1]);
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
 
//  ������һ����ע�͵�xxx12.22
//  if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') )		//������ͷ��ȡ
    //      capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );	//��ʼ������ͷ��׽��
    capture = cvCaptureFromAVI(videoFile);

  cvNamedWindow( "result", 1 );  //��������
 
  if( capture )    //����Ƶ�ļ�(����ͷ)��֡����
    {
      for(frameNum = 0;;frameNum++)
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
//	  is_speaking( people, result[frameNum] );
	  if( cvWaitKey( 10 ) >= 0 )          //�ȴ�����10ms
	    break;
        }
      cvReleaseImage( &frame_copy );			//�ͷ�֡�����ڴ�ռ�
      cvReleaseCapture( &capture );			//�ͷŲ�׽����ͬʱ�ͷ�frame
    }
  cvDestroyWindow("result");		//���ٴ���
  beautify_result(result);
  is_speaking(people, result);
  writeToFile();
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
      int facesNum = (faces ? faces->total : 0);
	  facesNum = (facesNum > 10) ? 10 : facesNum;
	  for( i = 0; i < facesNum; i++ )
        {
	  CvRect* r = (CvRect*)cvGetSeqElem( faces, i );		//��faces���ݴ�CvSeqתΪCvRect ?
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

int xcorr(int a[], int b[], int Length, int k)//����غ���
{
	int val = 0;
	if (k >= Length || k <= -Length)
	{
		val = -1;
	}
	else
	{
		for (int i = 0; i <= Length - 1 - abs(k); i ++)
		{
			if (k >= 0)
			{
				val = val + b[i] * a[i + k];
			}
			else
			{
				val = val + a[i] * b[i - k];
			}
		}
	}
	return val;
}

int is_speaking(int people ,FaceInfo result[FRAME_NUM_MAX][10])
{

	FILE* audio[4];
	for (int i = 0; i <= 3; i++)
	{
		audio[i] = fopen(audioFile[i], "r");
		if (audio[i] == NULL)
		{
			printf("cannot open file : %s", audioFile);
			exit(0);
		}
	}
	int data;
	double powerAvg = 0;	//��ƽ������
	for (int i = 0; i <= audioDataNum - 1; i ++)
	{
		fscanf(audio[0], "%d", &data);
		powerAvg = powerAvg + data * data;
	}
	powerAvg = powerAvg / audioDataNum;
	rewind(audio[0]);
	int jmax;
	int cnt = 0;
	double audioDataNumPerFrame = double(audioDataNum) / totalFrame;
	int** audioData = new int*[4];
	for (int i = 0; i <= 3; i ++)
	{
		audioData[i] = new int[int(audioDataNumPerFrame) + 1];
	}
	for (int i = 0; i <= totalFrame - 1; i ++)
	{
		jmax = int(audioDataNumPerFrame);
		if (jmax + cnt < audioDataNumPerFrame * (i + 1) - 1) //������ܳ��ֵ�audioDataNumPerFrame���������⡣
		{
			jmax = jmax + 1;
		}
		for (int k = 0; k <= 3; k ++)
		{
			for (int j = 0; j <= jmax - 1; j ++)
			{
				fscanf(audio[k], "%d", &audioData[k][j]);
			}	
			cnt++;
		}
		if (jmax == int(audioDataNumPerFrame))
		{
			for (int k = 0; k <= 3; k ++)
			{
				audioData[k][jmax] = 0;
			}
		}
		double corrMax[3] = {0};
		int argMaxCorr[3] = {0};
		point3 t_delay;
		double powerFrameAvg = 0;//��֡��������
		for(int j = 0; j <= 2; j++)//��3�������ļ�ѭ��
		{	
			for (int k = - int(audioDataNumPerFrame) + 1; k <= int(audioDataNumPerFrame) - 1; k++)
			{
				double corr = (double)xcorr(audioData[0], audioData[j + 1], int(audioDataNumPerFrame) + 1, k);
				if (corr > corrMax[j])
				{
					corrMax[j] = corr;
					argMaxCorr[j] = k;
				}
			}
			powerFrameAvg = powerFrameAvg + corrMax[j] / (jmax - argMaxCorr[j]); 
		}
		t_delay.x = - double(argMaxCorr[0]) / fs;
		t_delay.y = - double(argMaxCorr[1]) / fs;
		t_delay.z = - double(argMaxCorr[2]) / fs;
		powerFrameAvg = powerFrameAvg / 3;
		double thresholdPower = 0;//powerAvg / 10000;
		double thresholdDelay = 3e-4;
		point3 t_delay_idea;
		if (powerFrameAvg >= thresholdPower)
		{
			bool finish = false;
			for (int j = 0; j <= people - 1; j++)
			{
				t_delay_idea.x = (distanceP3(speakerLoc[j], micLoc[1]) - distanceP3(speakerLoc[j], micLoc[0])) / V_SOUND; 
				t_delay_idea.y = (distanceP3(speakerLoc[j], micLoc[2]) - distanceP3(speakerLoc[j], micLoc[0])) / V_SOUND; 
				t_delay_idea.z = (distanceP3(speakerLoc[j], micLoc[3]) - distanceP3(speakerLoc[j], micLoc[0])) / V_SOUND; 
				if (distanceP3(t_delay, t_delay_idea) <= thresholdDelay && !finish)
				{
					result[i][j].isSpeaking = 1;
					finish = true;
				}
				else
				{
					result[i][j].isSpeaking = 0;
				}
			}
		}
		else
		{
			for (int j = 0; j <= people - 1; j++)
			{
				result[i][j].isSpeaking = 0;
			}
		}
	}
	for (int i = 0; i <= 3; i++)
	{
		delete[] audioData[i];
		fclose(audio[i]);
	}
	delete audioData;
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
  point3 camLoc;
  fscanf(fileparam, "%d%d%d", &num_video, &num_audio, &people);
  fscanf(fileparam, "%d", &dataLength);
  fscanf(fileparam, "%d%d", &fps, &fs);
  totalFrame = fps * dataLength;
  audioDataNum = fs * dataLength;
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
  qsort(speakerLoc, people, sizeof(speakerLoc[0]), pLeftThan);//�������
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

void writeToFile()
{
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
