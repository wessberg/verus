#pragma once
// Audio device enumeration.

#include "common.hpp"

// ───────────────────────── DEVICE ENUM ─────────────────────────
struct DeviceInfo {
  int devId;
  std::string name;
  int maxInputChannels;
  int maxOutputChannels;
  double defaultSampleRate;
};

static std::vector<DeviceInfo> getDevices() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "Pa_Initialize failed: " << Pa_GetErrorText(err) << "\n";
    std::exit(1);
  }
  int n = Pa_GetDeviceCount();
  if (n < 0) {
    std::cerr << "Pa_GetDeviceCount failed: " << Pa_GetErrorText(n) << "\n";
    Pa_Terminate(); std::exit(1);
  }
  std::vector<DeviceInfo> devices;
  devices.reserve(n);
  for (int i = 0; i < n; ++i) {
    const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
    if (!info) continue;
    DeviceInfo d;
    d.devId = i;
    d.name = info->name ? info->name : "";
    d.maxInputChannels  = info->maxInputChannels;
    d.maxOutputChannels = info->maxOutputChannels;
    d.defaultSampleRate = info->defaultSampleRate;
    devices.push_back(d);
  }
  Pa_Terminate();
  return devices;
}

static void listDevices(const std::vector<DeviceInfo> &devices) {
  std::cout<<"PortAudio devices:\n";
  for (const auto &d : devices) {
    std::cout<<"  "<<d.devId<<": "<<d.name
             <<" (inputs: "<<d.maxInputChannels
             <<", outputs: "<<d.maxOutputChannels
             <<", sampleRate: "<<d.defaultSampleRate<<")\n";
  }
}
