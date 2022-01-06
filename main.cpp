#include <iostream>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

int main(int argc, char *argv[]) {
    auto logger = spdlog::stdout_color_mt("console");
    spdlog::set_default_logger(logger);
    spdlog::info("Started logger");

    if (argc != 2) {
        spdlog::error("Usage: <left camera index> <left camera index>");
        return -1;
    }

    auto leftCap = cv::VideoCapture(std::stoi(argv[0]));
    auto rightCap = cv::VideoCapture(std::stoi(argv[1]));

    return 0;
}

void get_depth_map(cv::Mat &imgL, cv::Mat &imgR) {
    auto window_size = 3;
    auto left_matcher = cv::StereoSGBM::create(-1, 5*16, window_size,
                                               8 * 3 * window_size, 32 * 3 * window_size, 12,
                                               63, 10, 50, 32,
                                               cv::StereoSGBM::MODE_SGBM_3WAY);

    auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    auto lmbda = 80000;
    auto sigma = 1.3;
    auto visual_multiplier = 6;

    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    wls_filter->setLambda(lmbda);
    wls_filter->setSigmaColor(sigma);
    auto displ = cv::Mat(imgL.rows, imgL.cols, CV_16S);
    auto dispr = cv::Mat(imgL.rows, imgL.cols, CV_16S);
    left_matcher->compute(imgL, imgR, displ);
    right_matcher->compute(imgR, imgL, dispr);

    wls_filter->filter(displ, imgL, dispr, imgR);
}
