#include "stdafx.h"
#include "opencv2/core/core.hpp"
using namespace cv;
#include<math.h>
 void solvec(RotatedRect& rect,float *c);
 void crossproduct(float *a,float*b,float*n);//n=a*b
 void solvec(RotatedRect& rect,float *c)
 {
	 float angle=rect.angle*3.14159/180.0;
	 float cosa=cos(angle),sina=sin(angle);
	 float semi_w=rect.size.width/2.0,semi_h=rect.size.height/2.0;
	 c[0]=pow(semi_h*cosa,2)+pow(semi_w*sina,2);
	 c[1]=pow(semi_h*sina,2)+pow(semi_w*cosa,2);
	 c[2]=sin(2*angle)*(pow(semi_h,2)-pow(semi_w,2));
	 c[3]=pow(semi_w*semi_h,2);
 }
 void crossproduct(float *a,float*b,float*n)
 {
	 a[0]*=100;a[1]*=100;a[2]*=100;
     b[0]*=100;b[1]*=100;b[2]*=100;//把a，b单位转换为cm
	 n[0]=a[1]*b[2]-a[2]*b[1];
	 n[1]=a[2]*b[0]-a[0]*b[2];
	 n[2]=a[0]*b[1]-a[1]*b[0];
	 float lenth=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
     n[0]=n[0]*0.1/lenth;
     n[1]=n[1]*0.1/lenth;
	 n[2]=n[2]*0.1/lenth;//把法向量归一化为0.2m

 }