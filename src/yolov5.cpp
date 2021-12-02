// tensorrtx/yolov5
#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "preprocess.h"

// ros
#include <ros/ros.h>
#include <stdio.h>
#include <string>
#include <cstdio>
#include <algorithm>
#include <time.h>
#include <vector>

#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <std_msgs/Header.h>
#include <std_msgs/Int32.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

// tensorrtx/yolov5
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

// globalize yolov5 variables for callback
int inputIndex;
int outputIndex;
uint8_t* img_host;
uint8_t* img_device;
cudaStream_t stream;
float* buffers[2];
static float prob[BATCH_SIZE * OUTPUT_SIZE];
IExecutionContext* context;

// ros
ros::Publisher pub_image;
ros::Subscriber sub_image;
int fcount = 0;


void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

void imageCallback(const sensor_msgs::Image::ConstPtr& raw_image) {
    fcount++;

    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(raw_image, sensor_msgs::image_encodings::BGR8);
    } catch(cv_bridge::Exception& e) {
	throw std::invalid_argument("CV_BRIDGE EXCEPTION");
    }

    cv::Mat frame = cv_ptr->image;

    //auto start = std::chrono::system_clock::now();
    float* buffer_idx = (float*)buffers[inputIndex];
    size_t size_image = frame.cols * frame.rows * 3;
    size_t size_image_dst = INPUT_H * INPUT_W * 3;

    //copy data to pinned memory
    memcpy(img_host,frame.data,size_image);

    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
    preprocess_kernel_img(img_device, frame.cols, frame.rows, buffer_idx, INPUT_W, INPUT_H, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << "[Plate Detection]: inference time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::vector<std::vector<Yolo::Detection>> batch_res(1);

    auto& res = batch_res[0];
    nms(res, &prob[0], CONF_THRESH, NMS_THRESH);

    cv::Mat img = cv_ptr->image;
    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(img, res[j].bbox);

        cv_bridge::CvImage img_bridge;
        sensor_msgs::Image img_msg;

        std_msgs::Header header = raw_image->header;
        img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, img(r));
        img_bridge.toImageMsg(img_msg);
	pub_image.publish(img_msg);
	    
        cv::imwrite("/root/catkin_ws/src/plate_detection/src/" + std::to_string(fcount) + ".jpg", frame(r));
    }
}

int main(int argc, char** argv) {
    std::cout << "[Plate Detection]: Cuda Load Start" << std::endl;

    cudaSetDevice(DEVICE);

    std::string engine_name = "/root/catkin_ws/src/plate_detection/src/yolov5m.engine";

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));
    img_host = nullptr;
    img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    std::cout << "[Plate Detection]: Cuda Load Finish" << std::endl;

    ros::init(argc, argv, "plate_detection_node");
    ros::NodeHandle nh("");

    pub_image = nh.advertise<sensor_msgs::Image>("/crop_image", 5);
    sub_image = nh.subscribe<sensor_msgs::Image>("/usb_cam/image_raw", 5, imageCallback);

    ros::spin();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();


    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
