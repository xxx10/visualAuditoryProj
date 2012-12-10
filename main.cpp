#include "opencv/cv.h"
#include "opencv/highgui.h"
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
 
#ifdef _EiC
#define WIN32
#endif
 
static CvMemStorage* storage = 0;		//storage �ڴ�飬ȫ�ֱ���
static CvHaarClassifierCascade* cascade = 0;		//������
 
void detect_and_draw( IplImage* image );
 
const char* cascade_name = "haarcascade_frontalface_alt.xml";		//������·��
/*    "haarcascade_profileface.xml";*/
 
int main( int argc, char** argv )
{
    CvCapture* capture = 0;
    IplImage *frame, *frame_copy = 0;
    int optlen = strlen("--cascade="); //?  --cascade=Ϊ������ѡ��ָʾ���� ?
    const char* input_name;
    
        cascade_name = "haarcascade_frontalface_alt2.xml";   //���ط�����
        //opencvװ�ú�haarcascade_frontalface_alt2.xml��·��,
       //Ҳ���԰�����ļ�������Ĺ����ļ�����Ȼ����д·����cascade_name= "haarcascade_frontalface_alt2.xml";  
       //����cascade_name ="C:\\Program Files\\OpenCV\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"
        input_name = argc > 1 ? argv[1] : 0;
    
 
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
 
    if( !cascade )		//���ط���������
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        fprintf( stderr,
        "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
    }
    storage = cvCreateMemStorage(0);		// �����ڴ�洢�����ڴ��
 
    if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') )		//������ͷ��ȡ
        capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );	//��ʼ������ͷ��׽��
 
    cvNamedWindow( "result", 1 );  //��������
 
    if( capture )    //����Ƶ�ļ�(����ͷ)��֡����
    {
        for(;;)
        {
            if( !cvGrabFrame( capture ))		//����ͷ����Ϣ������
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
 
            if( cvWaitKey( 10 ) >= 0 )          //�ȴ�����10ms
                break;
        }
 
        cvReleaseImage( &frame_copy );			//�ͷ�֡�����ڴ�ռ�
        cvReleaseCapture( &capture );			//�ͷ�����ͷ��׽����ͬʱ�ͷ�frame
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
 
    double scale = 1.3;
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),
                         cvRound (img->height/scale)),
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
        t = (double)cvGetTickCount() - t;		//ͳ�Ƽ��ʱ��
        printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );		//��faces���ݴ�CvSeqתΪCvRect ?
            CvPoint center;
            int radius;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
        }
		//t = (double)cvGetTickCount() - t;		//ͳ�Ƽ�⼰����ʱ��
		//printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
    }
 
    cvShowImage( "result", img );
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}
