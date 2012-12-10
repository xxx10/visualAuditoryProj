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
 
static CvMemStorage* storage = 0;		//storage 内存块，全局变量
static CvHaarClassifierCascade* cascade = 0;		//分类器
 
void detect_and_draw( IplImage* image );
 
const char* cascade_name = "haarcascade_frontalface_alt.xml";		//分类器路径
/*    "haarcascade_profileface.xml";*/
 
int main( int argc, char** argv )
{
    CvCapture* capture = 0;
    IplImage *frame, *frame_copy = 0;
    int optlen = strlen("--cascade="); //?  --cascade=为分类器选项指示符号 ?
    const char* input_name;
    
        cascade_name = "haarcascade_frontalface_alt2.xml";   //加载分类器
        //opencv装好后haarcascade_frontalface_alt2.xml的路径,
       //也可以把这个文件拷到你的工程文件夹下然后不用写路径名cascade_name= "haarcascade_frontalface_alt2.xml";  
       //或者cascade_name ="C:\\Program Files\\OpenCV\\data\\haarcascades\\haarcascade_frontalface_alt2.xml"
        input_name = argc > 1 ? argv[1] : 0;
    
 
    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
 
    if( !cascade )		//加载分类器出错
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        fprintf( stderr,
        "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
    }
    storage = cvCreateMemStorage(0);		// 创建内存存储器，内存块
 
    if( !input_name || (isdigit(input_name[0]) && input_name[1] == '\0') )		//从摄像头读取
        capture = cvCaptureFromCAM( !input_name ? 0 : input_name[0] - '0' );	//初始化摄像头捕捉器
 
    cvNamedWindow( "result", 1 );  //创建窗口
 
    if( capture )    //对视频文件(摄像头)逐帧处理
    {
        for(;;)
        {
            if( !cvGrabFrame( capture ))		//摄像头无信息，跳出
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
 
            if( cvWaitKey( 10 ) >= 0 )          //等待键盘10ms
                break;
        }
 
        cvReleaseImage( &frame_copy );			//释放帧副本内存空间
        cvReleaseCapture( &capture );			//释放摄像头捕捉器，同时释放frame
    }
 
    cvDestroyWindow("result");		//销毁窗口
 
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
        t = (double)cvGetTickCount() - t;		//统计监测时间
        printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );		//将faces数据从CvSeq转为CvRect ?
            CvPoint center;
            int radius;
            center.x = cvRound((r->x + r->width*0.5)*scale);
            center.y = cvRound((r->y + r->height*0.5)*scale);
            radius = cvRound((r->width + r->height)*0.25*scale);
            cvCircle( img, center, radius, colors[i%8], 3, 8, 0 );
        }
		//t = (double)cvGetTickCount() - t;		//统计监测及绘制时间
		//printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
    }
 
    cvShowImage( "result", img );
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}
