#include "opencv2/flann.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2\opencv.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
//#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\imgproc\types_c.h>
//#include <opencv2\objdetect\objdetect_c.h>

#include <stdlib.h>
#include <math.h>
#include <vector>
using namespace cv;
using namespace std;
#define PI 3.1415926535
#define RADIAN(angle) ((angle)*PI/180.0)

#define INVALDE_DEM_VALUE  -9999999

double amax = 0, Radius =1;


double GetDistence(double x, double y, double xm, double ym)
{
	return sqrt(pow((x - xm), 2) + pow((y - ym), 2));
}

double cubic_coeff(double x) {
	x = (x > 0) ? x : -x;
	if (x < 1) {
		return 1 - 2 * x*x + x * x*x;
	}
	else if (x < 2) {
		return 4 - 8 * x + 5 * x*x - x * x*x;
	}
	return 0;
}

Mat changeShapeResampleBilinear(const Mat &src) {
	Mat imgAffine = Mat::zeros(src.rows, src.cols, src.type());
	int row = imgAffine.rows, col = imgAffine.cols;
	double xm = src.rows / 2.0;
	double ym = src.cols / 2.0;
	
	
	for (int x = 0; x < row; x++) {
		for (int y = 0; y < col; y++) {

			double X = x / ((row - 1) / 2.0) - 1.0;
			double Y = y / ((col - 1) / 2.0) - 1.0;
			double r = sqrt(X * X + Y * Y);

			if (r >= Radius/(row/2)) {
				imgAffine.at<Vec3b>(x, y)[0] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[0]);
				imgAffine.at<Vec3b>(x, y)[1] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[1]);
				imgAffine.at<Vec3b>(x, y)[2] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[2]);
			}
			else {

				double theta = RADIAN(amax)*(Radius - GetDistence(x, y, xm, ym)) / Radius;

				//double theta = 1.0 + X * X + Y * Y - 2.0*sqrt(X * X + Y * Y);//修改不用（1-r)*(1-r)
				double x_ = cos(theta)*X - sin(theta)*Y;
				double y_ = sin(theta)*X + cos(theta)*Y;

				x_ = (x_ + 1.0)*((row - 1) / 2.0);
				y_ = (y_ + 1.0)*((col - 1) / 2.0);


				if (x_ < 0 || y_ < 0 || x_ >= src.rows || y_ >= src.cols) {
					for (int c = 0; c < 3; c++) {
						imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(0);
					}
				}
				else {
					//左上角坐标（X1，Y1)
					//计算双线性插值   
					int X1 = (int)x_;
					int Y1 = (int)y_;

					for (int c = 0; c < 3; c++) {
						if (X1 == (src.rows - 1) || Y1 == (src.cols - 1)) {
							imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(src.at<Vec3b>(X1, Y1)[c]);
						}
						else {
							//四个顶点像素值
							//注意访问越界
							int aa = src.at<Vec3b>(X1, Y1)[c];
							int bb = src.at<Vec3b>(X1, Y1 + 1)[c];
							int cc = src.at<Vec3b>(X1 + 1, Y1)[c];
							int dd = src.at<Vec3b>(X1 + 1, Y1 + 1)[c];

							double dx = x_ - (double)X1;
							double dy = y_ - (double)Y1;
							double h1 = aa + dx * (bb - aa);
							double h2 = cc + dx * (dd - cc);
							imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(h1 + dy * (h2 - h1));
						}
					}
				}
			}
		}
	}
	return imgAffine;
}


/**
* @brief 三次卷积
*
* @param x 插值点列坐标
* @param y 插值点行坐标
* @param pdfValue 影像数据
* @param nWidth 影像数据宽度
* @param nHeight 影像数据高度
*/
double ResampleCubic(double x, double y, double * pdfValue, int nWidth, int nHeight)
{
	double dfCubicValue;
	int i = x;
	int j = y;

	/*we do not handle the border, attention*/
	if ((i - 1) < 0 || (j - 1) < 0 || (j + 2) > (nHeight - 1) || (i + 2) > (nWidth - 1))
		return INVALDE_DEM_VALUE;

	/*get adjacent 16 values*/
	double values[4][4];
	for (int r = j - 1, s = 0; r <= j + 2; r++, s++) {
		for (int c = i - 1, t = 0; c <= i + 2; c++, t++) {
			values[s][t] = pdfValue[r*nWidth + c];
		}
	}

	/*calc the coeff*/
	double u = x - i;
	double v = y - j;
	double A[4], C[4];
	for (int distance = 1, s = 0; distance >= -2; distance--, s++) {
		A[s] = cubic_coeff(u + distance);
		C[s] = cubic_coeff(v + distance);
	}

	dfCubicValue = 0;
	for (int s = 0; s < 4; s++) {
		for (int t = 0; t < 4; t++) {
			dfCubicValue += values[s][t] * A[t] * C[s];
		}
	}
	return dfCubicValue;
}


float BiCubicPloy(float x)
{
	float abs_x = abs(x);//取x的绝对值
	float a = -0.5;
	if (abs_x <= 1.0)
	{
		return (a + 2)*pow(abs_x, 3) - (a + 3)*pow(abs_x, 2) + 1;
	}
	else if (abs_x < 2.0)
	{
		return a * pow(abs_x, 3) - 5 * a*pow(abs_x, 2) + 8 * a*abs_x - 4 * a;
	}
	else
		return 0.0;
}



Mat changeShapeResampleCubic(const Mat &src) {
	Mat imgAffine = Mat::zeros(src.rows, src.cols, src.type());
	int row = imgAffine.rows, col = imgAffine.cols;
	double xm = src.rows / 2.0;
	double ym = src.cols / 2.0;
	Mat g = src;

	

	for (int x = 0; x < row; x++) {
		

		for (int y = 0; y < col; y++) {

			double X = x / ((row - 1) / 2.0) - 1.0;
			double Y = y / ((col - 1) / 2.0) - 1.0;
			double r = sqrt(X * X + Y * Y);

			if (r >= Radius / (row / 2)) {
				imgAffine.at<Vec3b>(x, y)[0] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[0]);
				imgAffine.at<Vec3b>(x, y)[1] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[1]);
				imgAffine.at<Vec3b>(x, y)[2] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[2]);
			}
			else {

				double theta = RADIAN(amax)*(Radius - GetDistence(x, y, xm, ym)) / Radius;

				//double theta = 1.0 + X * X + Y * Y - 2.0*sqrt(X * X + Y * Y);//修改不用（1-r)*(1-r)
				double x_ = cos(theta)*X - sin(theta)*Y;
				double y_ = sin(theta)*X + cos(theta)*Y;

				x_ = (x_ + 1.0)*((row - 1) / 2.0);
				y_ = (y_ + 1.0)*((col - 1) / 2.0);


				if (x_ < 0 || y_ < 0 || x_ >= src.rows || y_ >= src.cols) {
					for (int c = 0; c < 3; c++) {
						imgAffine.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(0);
					}
				}
				else {
					//左上角坐标（X1，Y1)
					//计算双三次插值   
					int X1 = (int)x_;
					int Y1 = (int)y_;
					
					//获取目标图像(i,j)在原图中的坐标
					int xm = X1 ;
					int ym = Y1 ;

					//取出映射到原图的整数部分
					int xi = (int)xm;
					int yi = (int)ym;

					//取出映射到原图中的点的四周的16个点的坐标
					int x0 = xi - 1;
					int y0 = yi - 1;
					int x1 = xi;
					int y1 = yi;
					int x2 = xi + 1;
					int y2 = yi + 1;
					int x3 = xi + 2;
					int y3 = yi + 2;
					if ((x0 >= 0) && (x3 < src.rows) && (y0 >= 0) && (y3 < src.cols))
					{
						//求出行和列所对应的系数
						float dist_x0 = BiCubicPloy(xm - x0);
						float dist_x1 = BiCubicPloy(xm - x1);
						float dist_x2 = BiCubicPloy(xm - x2);
						float dist_x3 = BiCubicPloy(xm - x3);
						float dist_y0 = BiCubicPloy(ym - y0);
						float dist_y1 = BiCubicPloy(ym - y1);
						float dist_y2 = BiCubicPloy(ym - y2);
						float dist_y3 = BiCubicPloy(ym - y3);

						//k_i_j=k_i*k_j
						float dist_x0y0 = dist_x0 * dist_y0;
						float dist_x0y1 = dist_x0 * dist_y1;
						float dist_x0y2 = dist_x0 * dist_y2;
						float dist_x0y3 = dist_x0 * dist_y3;
						float dist_x1y0 = dist_x1 * dist_y0;
						float dist_x1y1 = dist_x1 * dist_y1;
						float dist_x1y2 = dist_x1 * dist_y2;
						float dist_x1y3 = dist_x1 * dist_y3;
						float dist_x2y0 = dist_x2 * dist_y0;
						float dist_x2y1 = dist_x2 * dist_y1;
						float dist_x2y2 = dist_x2 * dist_y2;
						float dist_x2y3 = dist_x2 * dist_y3;
						float dist_x3y0 = dist_x3 * dist_y0;
						float dist_x3y1 = dist_x3 * dist_y1;
						float dist_x3y2 = dist_x3 * dist_y2;
						float dist_x3y3 = dist_x3 * dist_y3;

						imgAffine.at<Vec3b>(x, y) = (src.at<Vec3b>(x0, y0)*dist_x0y0 +
							src.at<Vec3b>(x0, y1)*dist_x0y1 +
							src.at<Vec3b>(x0, y2)*dist_x0y2 +
							src.at<Vec3b>(x0, y3)*dist_x0y3 +
							src.at<Vec3b>(x1, y0)*dist_x1y0 +
							src.at<Vec3b>(x1, y1)*dist_x1y1 +
							src.at<Vec3b>(x1, y2)*dist_x1y2 +
							src.at<Vec3b>(x1, y3)*dist_x1y3 +
							src.at<Vec3b>(x2, y0)*dist_x2y0 +
							src.at<Vec3b>(x2, y1)*dist_x2y1 +
							src.at<Vec3b>(x2, y2)*dist_x2y2 +
							src.at<Vec3b>(x2, y3)*dist_x2y3 +
							src.at<Vec3b>(x3, y0)*dist_x3y0 +
							src.at<Vec3b>(x3, y1)*dist_x3y1 +
							src.at<Vec3b>(x3, y2)*dist_x3y2 +
							src.at<Vec3b>(x3, y3)*dist_x3y3);
					}
				}
			}
		}
	}
	return imgAffine;
}



void main() {


	const char *filename = "cb.png";//图像路径 
	Mat pano = cv::imread(filename);
	//namedWindow("pano", 1);
	//imshow("pano", pano);
	//waitKey(0);

	//cout << "输入最大旋转角度" << endl;
	//cin >> amax;
	//cout << "输入扭曲旋转半径" << endl;
	//cin >> Radius;

	//Mat bilShow;
	//bilShow = pano;
	//bilShow = changeShapeResampleBilinear(bilShow);
	//namedWindow("bil", 1);
	//imshow("bil", bilShow);
	//

	//
	//Mat cubShow;
	//cubShow = pano;
	//cubShow = changeShapeResampleCubic(cubShow);
	//namedWindow("cub", 1);
	//imshow("cub", cubShow);


    long time = clock(); 
	Mat img = imread(filename);
	Mat drcimg(img.rows, img.cols, CV_8UC3);
	cv::imshow("矫正前", img);

	cv::Point lenscenter(img.cols / 2, img.rows / 2);//镜头中心在图像中的位置  
	CvPoint src_a, src_b, src_c, src_d;//a、b、c、d四个顶点
	//矫正参数
	double r;//矫正前像素点跟镜头中心的距离
	double s;//矫正后像素点跟镜头中心的距离
	CvPoint2D32f mCorrectPoint;//矫正后点坐标
	double distance_to_a_x, distance_to_a_y;//求得中心点和边界的距离
	
	double c0 = -0.9998;
	double c1 = -4.3932;
	double c2 = +1.4327; //inner outer
	double c3 = -2.8526;
	double c4 = +9.8223;

	for (int y = 0; y < img.rows; y++)//操作数据区,要注意OpenCV的RGB的存储顺序为GBR    
		for (int x = 0; x < img.cols; x++)//示例为亮度调节    
		{
			r = sqrt((y - lenscenter.y)*(y - lenscenter.y) + (x - lenscenter.x)*(x - lenscenter.x));
			s = c0 
				+c1*1e-4 * r//pow(r,1) 
				+c2*1e-6 * r * r 
				+c3*1e-9 * r * r * r 
				+c4*1e-13 * r * r * r * r;//比例  
			
			mCorrectPoint = cvPoint2D32f((x - lenscenter.x) / s * 1.35 + lenscenter.x, (y - lenscenter.y) / s * 1.35 + lenscenter.y);
			//越界判断
			if (mCorrectPoint.y < 0 || mCorrectPoint.y >= img.rows - 1)
			{
				continue;
			}
			if (mCorrectPoint.x < 0 || mCorrectPoint.x >= img.cols - 1)
			{
				continue;
			}
			src_a = cvPoint((int)mCorrectPoint.x, (int)mCorrectPoint.y);
			src_b = cvPoint(src_a.x + 1, src_a.y);
			src_c = cvPoint(src_a.x, src_a.y + 1);
			src_d = cvPoint(src_a.x + 1, src_a.y + 1);
			distance_to_a_x = mCorrectPoint.x - src_a.x;//在原图像中与a点的水平距离    
			distance_to_a_y = mCorrectPoint.y - src_a.y;//在原图像中与a点的垂直距离    	

			drcimg.at<Vec3b>(y, x)[0] =
				img.at<Vec3b>(src_a.y, src_a.x)[0] * (1 - distance_to_a_x)*(1 - distance_to_a_y) +
				img.at<Vec3b>(src_b.y, src_b.x)[0] * distance_to_a_x*(1 - distance_to_a_y) +
				img.at<Vec3b>(src_c.y, src_c.x)[0] * distance_to_a_y*(1 - distance_to_a_x) +
				img.at<Vec3b>(src_d.y, src_d.x)[0] * distance_to_a_y*distance_to_a_x;
			drcimg.at<Vec3b>(y, x)[1] =
				img.at<Vec3b>(src_a.y, src_a.x)[1] * (1 - distance_to_a_x)*(1 - distance_to_a_y) +
				img.at<Vec3b>(src_b.y, src_b.x)[1] * distance_to_a_x*(1 - distance_to_a_y) +
				img.at<Vec3b>(src_c.y, src_c.x)[1] * distance_to_a_y*(1 - distance_to_a_x) +
				img.at<Vec3b>(src_d.y, src_d.x)[1] * distance_to_a_y*distance_to_a_x;
			drcimg.at<Vec3b>(y, x)[2] =
				img.at<Vec3b>(src_a.y, src_a.x)[2] * (1 - distance_to_a_x)*(1 - distance_to_a_y) +
				img.at<Vec3b>(src_b.y, src_b.x)[2] * distance_to_a_x*(1 - distance_to_a_y) +
				img.at<Vec3b>(src_c.y, src_c.x)[2] * distance_to_a_y*(1 - distance_to_a_x) +
				img.at<Vec3b>(src_d.y, src_d.x)[2] * distance_to_a_y*distance_to_a_x;
		}
	cv::flip(drcimg, drcimg,1);
	cv::imwrite("矫正完成.bmp", drcimg);
	cv::imshow("矫正完成", drcimg);
	cv::waitKey(0);
	img.release();
	drcimg.release();


	waitKey();
	return;
}