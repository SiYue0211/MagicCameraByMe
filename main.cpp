#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;

//复古图（滤镜）
Mat VintageColor(Mat srcImage)
{
	int width = srcImage.cols;
	int heigh = srcImage.rows;
	Mat dstImage(srcImage.size(), CV_8UC3);
	for (int y = 0; y < heigh; y++)
	{
		uchar* P0 = srcImage.ptr<uchar>(y);
		uchar* P1 = dstImage.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			float B = P0[3 * x];
			float G = P0[3 * x + 1];
			float R = P0[3 * x + 2];
			float newB = 0.272*R + 0.534*G + 0.131*B;
			float newG = 0.349*R + 0.686*G + 0.168*B;
			float newR = 0.393*R + 0.769*G + 0.189*B;
			if (newB < 0)newB = 0;
			if (newB > 255)newB = 255;
			if (newG < 0)newG = 0;
			if (newG > 255)newG = 255;
			if (newR < 0)newR = 0;
			if (newR > 255)newR = 255;
			P1[3 * x] = (uchar)newB;
			P1[3 * x + 1] = (uchar)newG;
			P1[3 * x + 2] = (uchar)newR;
		}

	}
	return dstImage;
}

//连环画（滤镜）
Mat ComicStripColor(Mat srcImage)
{
	int width = srcImage.cols;
	int heigh = srcImage.rows;
	RNG rng;
	Mat dstImage(srcImage.size(), CV_8UC3);
	for (int y = 0; y < heigh; y++)
	{
		uchar* P0 = srcImage.ptr<uchar>(y);
		uchar* P1 = dstImage.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			float B = P0[3 * x];
			float G = P0[3 * x + 1];
			float R = P0[3 * x + 2];
			float newB = abs(B - G + B + R)*G / 256;
			float newG = abs(B - G + B + R)*R / 256;
			float newR = abs(G - B + G + R)*R / 256;
			if (newB < 0)newB = 0;
			if (newB > 255)newB = 255;
			if (newG < 0)newG = 0;
			if (newG > 255)newG = 255;
			if (newR < 0)newR = 0;
			if (newR > 255)newR = 255;
			P1[3 * x] = (uchar)newB;
			P1[3 * x + 1] = (uchar)newG;
			P1[3 * x + 2] = (uchar)newR;
		}

	}
	Mat gray;
	cvtColor(dstImage, gray, CV_BGR2GRAY);
	normalize(gray, gray, 255, 0, CV_MINMAX);
	return gray;


}

//熔铸（滤镜）
Mat Casting(Mat srcImage)
{
	int width = srcImage.cols;
	int height = srcImage.rows;
	Mat dstImage(srcImage.size(), CV_8UC3);
	for (int y = 0; y < height; y++)
	{
		uchar *imgP = srcImage.ptr<uchar>(y);
		uchar *dstP = dstImage.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			float b0 = imgP[3*x];
			float g0 = imgP[3 * x + 1];
			float r0 = imgP[3 * x + 2];

			float b = b0 * 128/(g0 + b0 + 1);
			float g = g0 * 128 / (r0 + b0 + 1);
			float r = r0 * 128 / (b0 + g0 + 1);

			b = (b > 255 ? 255 : (b < 0 ? 0 : b));
			g = (g > 255 ? 255 : (g < 0 ? 0 : g));
			r = (r > 255 ? 255 : (r < 0 ? 0 : r));

			dstP[3 * x] = (uchar)b;
			dstP[3 * x + 1] = (uchar)g;
			dstP[3 * x + 2] = (uchar)r;

		}
	}
	return dstImage;
}

//浮雕（滤镜）
Mat Emboss(Mat srcImage)
{
	int width = srcImage.cols;
	int height = srcImage.rows;
	Mat dstImage(srcImage.size(), CV_8UC3);
	//进行卷积，采用矩阵为[1,0,0;0,0,0;0,0,-1],第一行,最后一行,第一列和最后一列不做处理
	for (int y = 1; y < height - 1; y++)
	{
		uchar *srcImagePreLine = srcImage.ptr<uchar>(y - 1);
		uchar *srcImageCurrLine = srcImage.ptr<uchar>(y);
		uchar *srcImageNextLine = srcImage.ptr<uchar>(y + 1);
		
		uchar *dstP = dstImage.ptr<uchar>(y);
		for (int x = 1; x < width-1; x++)
		{
			//分别对b,g,r进行处理
			for (int i = 0; i < 3; i++)
			{
				int temp = srcImagePreLine[3 * (x - 1) + i] - srcImageNextLine[3 * (x + 1) + i] + 100;
				temp = (temp > 255 ? 255 : (temp < 0 ? 0 : temp));
				dstP[3 * x + i] = temp;
			}
			
		}
	}
	return dstImage;
}

//哈哈镜——扩张（滤镜）
Mat Enlarge_face(Mat srcImage)
{
	int width = srcImage.cols;
	int heigh = srcImage.rows;
	Mat grayImage;
	std::vector<Rect> faces;
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "haarcascade_frontalface_alt不能加载" << std::endl;
	}
	cvtColor(srcImage, grayImage, COLOR_RGB2GRAY);
	face_cascade.detectMultiScale(grayImage, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	//获取人脸的中心,只处理识别到的第一个人脸，没有想好，好几个脸怎么处理
	std::cout << faces[0].x << "\t" << faces[0].y << std::endl;
	std::cout << faces[0].width << "\t" << faces[0].height << std::endl;
	Point lf(faces[0].x, faces[0].y);
	Point center(faces[0].x + (faces[0].width / 2.0), faces[0].y + (faces[0].height/ 2.0));
	std::cout << center;
	
	Mat dstImage(srcImage.size(), CV_8UC3);
	srcImage.copyTo(dstImage);

	//放大  
	int R1 = sqrtf(width*width + heigh*heigh) / 2; //直接关系到放大的力度,与R1成正比;  

	for (int y = 0; y<heigh; y++)
	{
		uchar *img1_p = dstImage.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			int dis = norm(Point(x, y) - center);
			if (dis<R1)
			{
				int newX = (x - center.x)*dis / R1 + center.x;
				int newY = (y - center.y)*dis / R1 + center.y;

				img1_p[3 * x] = srcImage.at<uchar>(newY, newX * 3);
				img1_p[3 * x + 1] = srcImage.at<uchar>(newY, newX * 3 + 1);
				img1_p[3 * x + 2] = srcImage.at<uchar>(newY, newX * 3 + 2);
			}
		}
	}
	return dstImage;
}

//哈哈镜——扩张（滤镜）
Mat Enlarge(Mat srcImage)
{
	int width = srcImage.cols;
	int heigh = srcImage.rows;
	Point center(width / 2, heigh / 2);
	Mat dstImage(srcImage.size(), CV_8UC3);
	srcImage.copyTo(dstImage);
	//【1】放大
	int R1 = sqrtf(width*width + heigh*heigh) / 2; //直接关系到放大的力度,与R1成正比;

	for (int y = 0; y<heigh; y++)
	{
		uchar *img1_p = dstImage.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			int dis = norm(Point(x, y) - center);
			if (dis<R1)
			{
				int newX = (x - center.x)*dis / R1 + center.x;
				int newY = (y - center.y)*dis / R1 + center.y;

				img1_p[3 * x] = srcImage.at<uchar>(newY, newX * 3);
				img1_p[3 * x + 1] = srcImage.at<uchar>(newY, newX * 3 + 1);
				img1_p[3 * x + 2] = srcImage.at<uchar>(newY, newX * 3 + 2);
			}
		}
	}
	return dstImage;
}

//人脸检测
Mat FaceDetection(Mat srcImage)
{
	Mat grayImage;
	std::vector<Rect> faces;
	String facecascade_name = "haarcascade_frontalface_alt.xml";
	CascadeClassifier facecascade;
	if (!facecascade.load(facecascade_name))
	{
		std::cout << "haarcascade_frontalface_alt不能加载" << std::endl;
	}
	cvtColor(srcImage, grayImage, COLOR_RGB2GRAY);
	facecascade.detectMultiScale(grayImage, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(srcImage, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),Scalar(0,0,225),3,4,0);
	}
	return srcImage;
}

int main()
{
	Mat srcImage = imread("peopleluna.jpg");
	if (!srcImage.data)
	{
		std::cout << "原图片读取失败"<< std::endl;
	}
	Mat dstImage(srcImage.size(), CV_8UC3);
	namedWindow("结果图",WINDOW_NORMAL);
	//复古色
	//dstImage = VintageColor(srcImage);
	//imshow("复古色", dstImage);
	//连环画
	//dstImage = ComicStripColor(srcImage);
	//imshow("连环画色", dstImage);
	//熔铸
	//dstImage = Casting(srcImage);
	//imshow("熔铸", dstImage);
	//dstImage = Emboss(srcImage);
	//dstImage = Enlarge_face(srcImage);
	dstImage = FaceDetection(srcImage);
	imshow("结果图", dstImage);
	waitKey();


}