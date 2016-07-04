// pgrcont.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include "triclops.h"
#include "pgrflycapture.h"
#include "pgrflycapturestereo.h"
#include "pnmutils.h"
#include <math.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
void solvec(RotatedRect& rect,float *c);
void crossproduct(float *a,float*b,float*n);//n=a*b

// Macro to check, report on, and handle Triclops API error codes.
#define _HANDLE_TRICLOPS_ERROR( description, error )	\
{ \
	if( error != TriclopsErrorOk ) \
   { \
   printf( \
   "*** Triclops Error '%s' at line %d :\n\t%s\n", \
   triclopsErrorToString( error ), \
   __LINE__, \
   description );	\
   printf( "Press any key to exit...\n" ); \
   getchar(); \
   exit( 1 ); \
   } \
} \

// Macro to check, report on, and handle Digiclops API error codes.
//
#define _HANDLE_FLYCAPTURE_ERROR( description, error )	\
{ \
	if( error != FLYCAPTURE_OK ) \
   { \
   printf( \
   "*** Flycapture Error '%s' at line %d :\n\t%s\n", \
   flycaptureErrorToString( error ), \
   __LINE__, \
   description );	\
   printf( "Press any key to exit...\n" ); \
   getchar(); \
   exit( 1 ); \
   } \
} \


int _tmain(int argc, _TCHAR* argv[])
{
	TriclopsContext   triclops;
	TriclopsImage     refImage;
	TriclopsInput     triclopsInput;

	FlyCaptureContext	   flycapture;
	FlyCaptureImage	   flycaptureImage;
	FlyCapturePixelFormat   pixelFormat;

	TriclopsError     te;
	FlyCaptureError   fe;
    
	Mat LeftImg,RightImg,LeftbwImg,RightbwImg,tempImg;
	Mat LeftDst,RightDst;
	Mat LeftMask=Mat::zeros(602,802,CV_8UC1);
	
	LeftImg.create(600,800,CV_8UC1);
	LeftDst.create(600,800,CV_8UC3);
	RightImg.create(600,800,CV_8UC1);
	RightDst.create(600,800,CV_8UC3);
	
    namedWindow("leftImg",0);
	namedWindow("leftbwImg",0);
	namedWindow("leftdst",0);

	namedWindow("rightImg",0);
	namedWindow("rightbwImg",0);
	namedWindow("rightdst",0);

	int iMaxCols = 0;
	int iMaxRows = 0;

	char* szCalFile;

	// Open the camera
	fe = flycaptureCreateContext( &flycapture );
	_HANDLE_FLYCAPTURE_ERROR( "flycaptureCreateContext()", fe );

	// Initialize the Flycapture context
	fe = flycaptureInitialize( flycapture, 0 );
	_HANDLE_FLYCAPTURE_ERROR( "flycaptureInitialize()", fe );

	// Save the camera's calibration file, and return the path 
	fe = flycaptureGetCalibrationFileFromCamera( flycapture, &szCalFile );
	_HANDLE_FLYCAPTURE_ERROR( "flycaptureGetCalibrationFileFromCamera()", fe );

	// Create a Triclops context from the cameras calibration file
	te = triclopsGetDefaultContextFromFile( &triclops, szCalFile );
	_HANDLE_TRICLOPS_ERROR( "triclopsGetDefaultContextFromFile()", te );

	pixelFormat = FLYCAPTURE_MONO16;
	iMaxCols = 1024; 
	iMaxRows = 768;   

	// Start transferring images from the camera to the computer
	fe = flycaptureStartCustomImage( 
		flycapture, 3, 0, 0, iMaxCols, iMaxRows, 100, pixelFormat);
	_HANDLE_FLYCAPTURE_ERROR( "flycaptureStart()", fe );

	te=triclopsSetResolution( triclops, 600, 800);

	// Grab an image from the camera
	fe = flycaptureGrabImage2( flycapture, &flycaptureImage );
	_HANDLE_FLYCAPTURE_ERROR( "flycaptureGrabImage()", fe );
   
	// Extract information from the FlycaptureImage
	int imageCols = flycaptureImage.iCols;
	int imageRows = flycaptureImage.iRows;
	int imageRowInc = flycaptureImage.iRowInc;
	int iSideBySideImages = flycaptureImage.iNumImages;
	unsigned long timeStampSeconds = flycaptureImage.timeStamp.ulSeconds;
	unsigned long timeStampMicroSeconds = flycaptureImage.timeStamp.ulMicroSeconds;

	// Create buffers for holding the mono images
	unsigned char* rowIntMono =new unsigned char[ imageCols * imageRows * iSideBySideImages ];
    
	// Create a temporary FlyCaptureImage for preparing the stereo image
	FlyCaptureImage tempImage;
	tempImage.pData = rowIntMono;

	// Pointers to positions in the mono buffer that correspond to the beginning
	// of the red, green and blue sections
	unsigned char* redMono = NULL;
	unsigned char* greenMono = NULL;
	unsigned char* blueMono = NULL;
	redMono = rowIntMono;
	greenMono = redMono + imageCols;
	blueMono = redMono + imageCols;

	int blocksize=37,c=10;
	int ele_type=MORPH_RECT,ele_size=2; 
	Mat element = getStructuringElement( ele_type,
		Size( 2*ele_size + 1, 2*ele_size+1 ),Point( ele_size, ele_size ) );

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Point> approx;
	vector<RotatedRect> Lellipse;
	vector<RotatedRect> Rellipse;
	int num1=1,num2=1,num3=1;
	char name[50];
	float xl,yl,zl,xr,yr,zr,normvec[3];

	while (true)//不断循环采集图像
	{
		#pragma region grab&rectify
			   // Grab an image from the camera
				fe = flycaptureGrabImage2( flycapture, &flycaptureImage );
				_HANDLE_FLYCAPTURE_ERROR( "flycaptureGrabImage()", fe );
			
				// Convert the pixel interleaved raw data to row interleaved format
				fe = flycapturePrepareStereoImage( flycapture, flycaptureImage, &tempImage, NULL);
				_HANDLE_FLYCAPTURE_ERROR( "flycapturePrepareStereoImage()", fe );
			
				// Use the row interleaved images to build up an RGB TriclopsInput.  
				// An RGB triclops input will contain the 3 raw images (1 from each camera).
				te = triclopsBuildRGBTriclopsInput(
					imageCols, 
					imageRows, 
					imageRowInc,  
					timeStampSeconds, 
					timeStampMicroSeconds, 
					redMono, 
					greenMono, 
					blueMono, 
					&triclopsInput);
				_HANDLE_TRICLOPS_ERROR( "triclopsBuildRGBTriclopsInput()", te );
			  
				// Rectify the images
				te = triclopsRectify( triclops, &triclopsInput );
				_HANDLE_TRICLOPS_ERROR( "triclopsRectify()", te );
			
				// Retrieve the rectified image from the triclops context
				te = triclopsGetImage( triclops, TriImg_RECTIFIED, TriCam_RIGHT, &refImage );
				_HANDLE_TRICLOPS_ERROR( "triclopsGetImage()", te );
				memcpy(RightImg.data,refImage.data,600*800);
			
				te = triclopsGetImage( triclops, TriImg_RECTIFIED, TriCam_LEFT, &refImage );
				_HANDLE_TRICLOPS_ERROR( "triclopsGetImage()", te );
				memcpy(LeftImg.data,refImage.data,600*800);
		#pragma endregion grab&rectify

		//equalizeHist(LeftImg,LeftImg);
		//equalizeHist(RightImg,RightImg);
		GaussianBlur(LeftImg,LeftImg, Size(5,5 ),0,0);
		GaussianBlur(RightImg,RightImg,Size(5,5),0,0);
		adaptiveThreshold(LeftImg,LeftbwImg,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,blocksize,-10);
		adaptiveThreshold(RightImg,RightbwImg,255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,blocksize,-10);
		morphologyEx(LeftbwImg,LeftbwImg,MORPH_CLOSE,element);
		morphologyEx(RightbwImg,RightbwImg,MORPH_CLOSE,element);
		Lellipse.clear();
		Rellipse.clear();

		#pragma region process left
		// int maskflag=8+FLOODFILL_MASK_ONLY|FLOODFILL_FIXED_RANGE+(255<<8);
				//LeftMask.setTo(Scalar(0));
				LeftbwImg.copyTo(tempImg);   //处理左图
				findContours(tempImg,contours,hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

				if(0==contours.size()) continue;
				int idx = 0;
				LeftDst.setTo(Scalar(0,0,0));
				for( ; idx >= 0; idx = hierarchy[idx][0])
				{
					approxPolyDP(contours[idx], approx, 5, true);
					double area1= contourArea(approx);//逼近的多边形的面积
					if (area1<25000&&area1>150)
					{
						RotatedRect box =minAreaRect(approx);
						float area2=box.size.area();//逼近的多边形的最小外接矩形的面积
						float ratio=area1>area2?area2/area1:area1/area2;
						if (ratio>0.80)
						{
							int idx2=hierarchy[idx][2];//first child contour
							int maxlenth=0,lenth,maxidx=idx2;
							for (;idx2>=0;idx2=hierarchy[idx2][0])
							{
							  lenth=contours[idx2].size();
							  if (maxlenth<lenth)
							  {
								  maxlenth=lenth;
								  maxidx=idx2;
							  }

							}//for idx2寻找最长的子轮廓
							if (maxlenth>0)//说明有子轮廓
							{
							  int a1=contourArea(contours[maxidx]);
							  //Point2f center; float radius;
							  //minEnclosingCircle(contours[maxidx],center,radius);//最长子轮廓的最小外接圆
							  RotatedRect box1=fitEllipse(contours[maxidx]);//最长子轮廓的拟合椭圆
							  RotatedRect box2=minAreaRect(contours[maxidx]);//最长子轮廓的最小外接矩形
							  float sellipse=3.14*box1.size.area()/4.0;
							  float ratio2=a1>sellipse?sellipse/a1:a1/sellipse;
							  float ratio3=a1/box2.size.area();
						
							  if (ratio2>ratio3&&ratio2>0.8&&box1.size.height>box.size.height/5.0)//最长子轮廓满足圆的条件
							  {
								  Scalar color(255,0,0);
								  Lellipse.push_back(box1);//保存左拟合椭圆参数
								  //circle(LeftDst,box1.center,2,Scalar(0,255,0),1,8);
								  drawContours(LeftDst, contours, idx, color,1, 8, hierarchy,1);
								  //floodFill(LeftImg,LeftMask,box1.center,200,0,Scalar(25),Scalar(25),maskflag);
								  drawContours(LeftImg, contours, idx, color,1, 8, hierarchy,1);
							  }
		                   
							  if (ratio3>=ratio2&&ratio3>0.8&&fabs(box.angle-box2.angle)<30)//最长子轮廓满足矩形的条件
							  {
								  Scalar color(0,0,255);
								  drawContours(LeftDst, contours, idx, color,1, 8, hierarchy,1);
								  drawContours(LeftImg, contours, idx, color,1, 8, hierarchy,1);
							  }
							}//if (maxlenth>0)
							
						}//if ratio>0.8
						
					}//if area1
					
				}//for idx left
		#pragma endregion process left

		#pragma region process right
				RightbwImg.copyTo(tempImg);//处理右图
				findContours(tempImg, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
				// iterate through all the top-level contours
				if (0==contours.size())  continue;
				idx = 0;
				RightDst.setTo(Scalar(0,0,0));
				for( ; idx >= 0; idx = hierarchy[idx][0])
				{
					approxPolyDP(contours[idx], approx, 5, true);
					double area1= contourArea(approx);//逼近的多边形的面积
					if (area1<25000&&area1>150)
					{
						RotatedRect box =minAreaRect(approx);
						float area2=box.size.area();//逼近的多边形的最小外接矩形的面积
						float ratio=area1>area2?area2/area1:area1/area2;
						if (ratio>0.80)
						{
							int idx2=hierarchy[idx][2];//first child contour
							int maxlenth=0,lenth,maxidx=idx2;
							for (;idx2>=0;idx2=hierarchy[idx2][0])
							{
								lenth=contours[idx2].size();
								if (maxlenth<lenth)
								{
									maxlenth=lenth;
									maxidx=idx2;
								}

							}//for idx2寻找最长的子轮廓
							if (maxlenth>0)//说明有子轮廓
							{
								int a1=contourArea(contours[maxidx]);
								//Point2f center; float radius;
								//minEnclosingCircle(contours[maxidx],center,radius);//最长子轮廓的最小外接圆
								RotatedRect box1=fitEllipse(contours[maxidx]);//最长子轮廓的拟合椭圆
								RotatedRect box2=minAreaRect(contours[maxidx]);//最长子轮廓的最小外接矩形
								//float ratio2=a1/(3.14*radius*radius);
								float sellipse=3.14*box1.size.area()/4.0;
								float ratio2=a1>sellipse?sellipse/a1:a1/sellipse;
								float ratio3=a1/box2.size.area();

								if (ratio2>ratio3&&ratio2>0.8&&box1.size.height>box.size.height/5.0)//最长子轮廓满足圆的条件
								{
									Scalar color(255,0,0);
									Rellipse.push_back(box1);//保存右拟合椭圆参数
									//circle(RightDst,box1.center,2,Scalar(0,255,0),1,8);
									drawContours(RightDst, contours, idx, color,1, 8, hierarchy,1);
									drawContours(RightImg, contours, idx, color,1, 8, hierarchy,1);
								}

								if (ratio3>=ratio2&&ratio3>0.8&&fabs(box.angle-box2.angle)<30)//最长子轮廓满足矩形的条件
								{
									Scalar color(0,0,255);
									drawContours(RightDst,contours, idx, color,1, 8, hierarchy,1);
									drawContours(RightImg, contours, idx, color,1, 8, hierarchy,1);
								}
							}//if (maxlenth>0)

						}//if ratio>0.8

					}//if area1

				}//for idx right
		#pragma endregion process right
		   
		#pragma region match
				if (Lellipse.empty()||Rellipse.empty())//根据椭圆中心匹配圆并显示圆心坐标
				{
				  sprintf(name,"no matched circle center");
				  putText(LeftDst,name,Point(0,20), FONT_HERSHEY_PLAIN,1,Scalar(0,255,0));
				  putText(RightDst,name,Point(0,20),FONT_HERSHEY_PLAIN,1,Scalar(0,255,0));
				}
				else
				{
				  int i,j,l=Lellipse.size(),r=Rellipse.size();
				  for (i=0;i<l;i++)
				  {
					  for (j=0;j<r;j++)
					  {
						  float dy=Lellipse[i].center.y-Rellipse[j].center.y;
						  if (fabs(dy)<2)// dy<2表示leftcenter[i]和rightcenter[j]可以匹配
						  {
							  float disp=fabs(Lellipse[i].center.x-Rellipse[j].center.x);
							  sprintf(name,"left center x=%5.2fpixel,y=%5.2fpixel",Lellipse[i].center.x,Lellipse[i].center.y);
							  putText(LeftDst,name,Point(0,20*num1), FONT_HERSHEY_PLAIN,1,Scalar(0,255,0));//显示图像坐标
							  triclopsRCDFloatToXYZ(triclops,Lellipse[i].center.y,Lellipse[i].center.x,disp,&xl,&yl,&zl);
							  sprintf(name,"xL=%5.2fcm,yL=%5.2fcm,zL=%5.2fcm",xl*100,yl*100,zl*100);
							  putText(LeftDst,name,Point(400,20*num1), FONT_HERSHEY_PLAIN,1,Scalar(255,255,0));//显示空间坐标
							  circle(LeftDst,Lellipse[i].center,2,Scalar(0,255,0),1,8);

							  sprintf(name,"right center x=%5.2fpixel,y=%5.2fpixel",Rellipse[j].center.x,Rellipse[j].center.y);
							  putText(RightDst,name,Point(0,20*num1), FONT_HERSHEY_PLAIN,1,Scalar(0,255,0));//显示图像坐标
							  triclopsRCDFloatToXYZ(triclops,Rellipse[j].center.y,Rellipse[j].center.x,disp,&xr,&yr,&zr);
							  sprintf(name,"xR=%5.2fcm,yR=%5.2fcm,zR=%5.2fcm",xr*100,yr*100,zr*100);
							  putText(RightDst,name,Point(400,20*num1), FONT_HERSHEY_PLAIN,1,Scalar(255,255,0));//显示空间坐标
							  circle(RightDst,Rellipse[j].center,2,Scalar(0,255,0));
							  

				  #pragma region normal vector

				  //int y,y1,y2,ymax=0,ymin=600,size=Lellipse[i].size(),k，n=0;
				  float LC[4],RC[4],ydelta;
				  Point2f LP[4],RP[4];
				  solvec(Lellipse[i],LC);
				  ydelta=sqrt(LC[0]*LC[3]/(4*LC[0]*LC[1]-LC[2]*LC[2]) );
				  //求左图四个点的图像坐标
				  LP[0].y=Lellipse[i].center.y-ydelta;
				  LP[1].y=Lellipse[i].center.y-ydelta;
				  LP[2].y=Lellipse[i].center.y+ydelta;
				  LP[3].y=Lellipse[i].center.y+ydelta;
				  LP[0].x=(LC[2]*ydelta-sqrt(3*LC[0]*LC[3]) )/(2*LC[0])+Lellipse[i].center.x;
				  LP[1].x=(LC[2]*ydelta+sqrt(3*LC[0]*LC[3]) )/(2*LC[0])+Lellipse[i].center.x;
				  LP[2].x=(-LC[2]*ydelta-sqrt(3*LC[0]*LC[3]) )/(2*LC[0])+Lellipse[i].center.x;
				  LP[3].x=(-LC[2]*ydelta+sqrt(3*LC[0]*LC[3]) )/(2*LC[0])+Lellipse[i].center.x;
				  //显示左图中的四个点
				  circle(LeftDst,LP[0],2,Scalar(255,255,255));
				  circle(LeftDst,LP[1],2,Scalar(0,255,0));
				  circle(LeftDst,LP[2],2,Scalar(0,0,255));
				  circle(LeftDst,LP[3],2,Scalar(0,255,255));
				  solvec(Rellipse[j],RC);
				  float rdelta1=pow(RC[2]*(dy-ydelta),2)-4*RC[0]*(RC[1]*(dy-ydelta)*(dy-ydelta)-RC[3]);
				  float rdelta2=pow(RC[2]*(dy+ydelta),2)-4*RC[0]*(RC[1]*(dy+ydelta)*(dy+ydelta)-RC[3]);
				  //求右图四个点的图像坐标
				  RP[0].y=LP[0].y;
				  RP[1].y=LP[1].y;
				  RP[2].y=LP[2].y;
				  RP[3].y=LP[3].y;
				  RP[0].x=(-RC[2]*(dy-ydelta)-sqrt(rdelta1))/(2*RC[0])+Rellipse[j].center.x;
				  RP[1].x=(-RC[2]*(dy-ydelta)+sqrt(rdelta1))/(2*RC[0])+Rellipse[j].center.x;
				  RP[2].x=(-RC[2]*(dy+ydelta)-sqrt(rdelta2))/(2*RC[0])+Rellipse[j].center.x;
				  RP[3].x=(-RC[2]*(dy+ydelta)+sqrt(rdelta2))/(2*RC[0])+Rellipse[j].center.x;
				  //显示右图中的四个点
				  circle(RightDst,RP[0],2,Scalar(255,255,255));
				  circle(RightDst,RP[1],2,Scalar(0,255,0));
				  circle(RightDst,RP[2],2,Scalar(0,0,255));
				  circle(RightDst,RP[3],2,Scalar(0,255,255));
				  //pos四个点的xyz坐标，vec两个矢量的xyz坐标,normal法向量
				  float pos[4][3],vec[2][3],normal[3],tip[3],disparity,row,col;
				  for(int k=0;k<4;k++)
				  {
					  float dx=fabs(LP[k].x-RP[k].x);
					  triclopsRCDFloatToXYZ(triclops,RP[k].y,RP[k].x,dx,&pos[k][0],&pos[k][1],&pos[k][2]);
				  }
				 vec[0][0]=pos[3][0]-pos[0][0];vec[0][1]=pos[3][1]-pos[0][1];vec[0][2]=pos[3][2]-pos[0][2];
				 vec[1][0]=pos[2][0]-pos[1][0];vec[1][1]=pos[2][1]-pos[1][1];vec[1][2]=pos[2][2]-pos[1][2];
				 crossproduct(vec[1],vec[0],normal);
				 tip[0]=xr+normal[0];tip[1]=yr+normal[1];tip[2]=zr+normal[2];
				 triclopsXYZToRCD(triclops,tip[0],tip[1],tip[2],&row,&col,&disparity);
				 line(RightDst,Rellipse[j].center,Point2f(col,row),Scalar(0,0,255),1);
				 line(RightImg,Rellipse[j].center,Point2f(col,row),Scalar(255),1);
				 line(LeftDst,Lellipse[i].center,Point2f(col+disparity,row),Scalar(0,0,255),1);
				 line(LeftImg,Lellipse[i].center,Point2f(col+disparity,row),Scalar(255),1);
				 sprintf(name,"normal vector:%5.2fcm,%5.2fcm,%5.2fcm",normal[0]*100,normal[1]*100,normal[2]*100);
				 putText(RightDst,name,Point(0,20*num1+20), FONT_HERSHEY_PLAIN,1,Scalar(255,255,0));//显示法向量
				 normvec[0]=normal[0]; normvec[1]=normal[1]; normvec[2]=normal[2];
				  #pragma endregion normal vector

				  num1+=2;
						  }//if dy<2
					  }//for j
				  }//for i
				}//if
				num1=1;
		#pragma endregion match

		#pragma region imshow
			imshow("leftImg",LeftImg);
			imshow("leftbwImg",LeftbwImg);
			imshow("leftdst",LeftDst);
			//imshow("leftmask",LeftMask);
			imshow("rightImg",RightImg);
			imshow("rightbwImg",RightbwImg);
			imshow("rightdst",RightDst);
			char a=waitKey(20);
			if (27==a) break;
			switch (a)
			{
			case '=':
				blocksize+=6;
				printf("now blocksize is %d\n",blocksize);
				break;
			case '-':
				if (blocksize>10)
				{
					blocksize-=6;
					printf("now blocksize is %d\n",blocksize);
				}
				break;
			case's':
				sprintf(name,"Lgray%d.bmp",num2);
				imwrite(name,LeftImg);
				sprintf(name,"Lbw%d.bmp",num2);
				imwrite(name,LeftbwImg);
				num2++;
				break;
			case 'b'://保存坐标数据xr，yr,zr,normal vec
				FILE *fp; 
				fp=fopen("data.txt","at");
				if(fp) 
				{sprintf(name,"%d:xr=%5.2f,yr=%5.2f,zr=%5.2f\n",num3,xr*100,yr*100,zr*100);
				fputs(name,fp);
				sprintf(name,"normal vec %5.2f,%5.2f,%5.2f\n",normvec[0]*100,normvec[1]*100,normvec[2]*100);
				fputs(name,fp);
				fclose(fp);
				num3++;
				}
				break;
			case 'c':
				num3=1;
				break;
			}
	#pragma endregion imshow
				
	}//while(true)不断采集图像

    delete []rowIntMono;
	flycaptureStop(flycapture);
    flycaptureDestroyContext(flycapture);
	triclopsDestroyContext(triclops);

	return 0;
}