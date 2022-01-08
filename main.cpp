#include <iostream>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

struct Config;

void rectify(Config &config, cv::Mat &imgL, cv::Mat &imgR, cv::Mat &distL, cv::Mat &distR);
void get_depth_map(cv::Mat &imgL, cv::Mat &imgR, cv::Mat &dist);

typedef struct Config {
    cv::Mat K1;
    cv::Mat D1;
    cv::Mat K2;
    cv::Mat D2;
    cv::Mat R;
    cv::Mat T;
    cv::Mat E;
    cv::Mat F;
    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
} Config;


Config new_config(const char* stereo) {
    cv::FileStorage fsStereo(stereo, cv::FileStorage::READ);
    if (!fsStereo.isOpened()) {
        spdlog::error("Failed to open stereo file \"{}\"", stereo);
    } else {
        spdlog::debug("Opened configs");
    }

    auto K1 = fsStereo.operator[]("K1").mat();
    auto D1 = fsStereo.operator[]("D1").mat();
    auto K2 = fsStereo.operator[]("K2").mat();
    auto D2 = fsStereo.operator[]("D2").mat();
    auto R = fsStereo.operator[]("R").mat();
    auto T = fsStereo.operator[]("T").mat();
    auto E = fsStereo.operator[]("E").mat();
    auto F = fsStereo.operator[]("F").mat();
    auto R1 = fsStereo.operator[]("R1").mat();
    auto R2 = fsStereo.operator[]("R2").mat();
    auto P1 = fsStereo.operator[]("P1").mat();
    auto P2 = fsStereo.operator[]("P2").mat();
    auto Q = fsStereo.operator[]("Q").mat();

    fsStereo.~FileStorage();
    return Config {K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q};
}

void rectify(Config &config, cv::Mat &imgL, cv::Mat &imgR, cv::Mat &distL, cv::Mat &distR) {
    spdlog::trace("Creating mats");
    auto mapX = cv::Mat(imgL.rows, imgL.cols, CV_32FC1);
    auto mapY = cv::Mat(imgL.rows, imgL.cols, CV_32FC1);
    spdlog::trace("Making undistortion map left");
    cv::initUndistortRectifyMap(config.K1, config.D1, config.R1, config.P1, imgL.size(), CV_32FC1, mapX, mapY);
    spdlog::trace("Remapping left");
    cv::remap(imgL, distL, mapX, mapY, cv::BORDER_CONSTANT);
    spdlog::trace("Making undistortion map right");
    cv::initUndistortRectifyMap(config.K2, config.D2, config.R2, config.P2, imgR.size(), CV_32FC1, mapX, mapY);
    spdlog::trace("Remapping right");
    cv::remap(imgR, distR, mapX, mapY, cv::BORDER_CONSTANT);

    cv::cvtColor(distL, distL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(distR, distR, cv::COLOR_BGR2GRAY);

    cv::blur(distL, distL, cv::Size(3, 3));
    cv::blur(distR, distR, cv::Size(3, 3));
}

void get_depth_map(cv::Mat &imgL, cv::Mat &imgR, cv::Mat &dist) {
    auto window_size = 3;
    spdlog::trace("Creating left_matcher");
    auto left_matcher = cv::StereoSGBM::create(0, 10*16, 11,
                                               8 * 3 * 11 * 11, 32 * 3 * 11 * 11, -1,
                                               0, 10, 200, 3,
                                               cv::StereoSGBM::MODE_SGBM_3WAY);

    spdlog::trace("Creating right_matcher");
    auto right_matcher = cv::ximgproc::createRightMatcher(left_matcher);

    auto lmbda = 80000;
    auto sigma = 1.3;
    auto visual_multiplier = 6;

    spdlog::trace("Creating left_wls_filter");
    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(left_matcher);
    wls_filter->setLambda(lmbda);
    wls_filter->setSigmaColor(sigma);
    auto displ = cv::Mat(imgL.rows, imgL.cols, CV_16S);
    auto dispr = cv::Mat(imgL.rows, imgL.cols, CV_16S);
    spdlog::trace("Computing disparity");
    left_matcher->compute(imgL, imgR, displ);
    right_matcher->compute(imgR, imgL, dispr);

    spdlog::trace("Computing filtered disparity");
    wls_filter->filter(displ, imgL, displ, dispr);
    spdlog::trace("Normalizing output");
    cv::normalize(displ, dist, 255, 0, cv::NORM_MINMAX, CV_8U);
}

int main(int argc, char *argv[]) {
    auto logger = spdlog::stdout_color_mt("console");
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
    spdlog::info("Started logger");

    if (argc != 4) {
        spdlog::error("Usage: <left camera index> <right camera index> <stereo.yml>, got {}", argc);
        return -1;
    }

    spdlog::debug("Opening videos {} {}", argv[1], argv[2]);
    auto leftCap = cv::VideoCapture(argv[1]);
    auto rightCap = cv::VideoCapture(argv[2]);
    spdlog::debug("Opened videos");

    auto disp = cv::Mat(480, 640, CV_8UC3);
    auto left = cv::Mat(480, 640, CV_8UC3);
    auto right = cv::Mat(480, 640, CV_8UC3);

    spdlog::debug("Reading configs");
    Config config = new_config(argv[3]);

    spdlog::info("Set up variables");
    while (true) {
        spdlog::debug("Reading frames");
        leftCap >> left;
        rightCap >> right;

        cv::imshow("Left", left);
        cv::imshow("Right", right);

        spdlog::debug("Rectifying image");
        rectify(config, left, right, left, right);
        spdlog::debug("Getting depth map");
        get_depth_map(left, right, disp);

        cv::imshow("Depth", disp);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}
