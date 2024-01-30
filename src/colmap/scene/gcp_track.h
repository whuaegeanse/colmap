// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/util/logging.h"
#include "colmap/util/types.h"

#include <vector>

#include <Eigen/Core>

namespace colmap {

// Track class stores all observations of a 3D point.
struct GCPTrackElement {
  GCPTrackElement();
  GCPTrackElement(image_t image_id, const Eigen::Vector2d& image_xy);
  // The image in which the track element is observed.
  image_t image_id;

  // The point in the image that the track element is observed.
  Eigen::Vector2d image_xy;
};

class GCPTrack {
 public:
  GCPTrack();

  // The number of track elements.
  inline size_t Length() const;

  // Access all elements.
  inline const std::vector<GCPTrackElement>& Elements() const;
  inline std::vector<GCPTrackElement>& Elements();
  inline void SetElements(std::vector<GCPTrackElement> elements);

  // Access specific elements.
  inline const GCPTrackElement& Element(size_t idx) const;
  inline GCPTrackElement& Element(size_t idx);
  inline void SetElement(size_t idx, const GCPTrackElement& element);

  // Append new elements.
  inline void AddElement(const GCPTrackElement& element);
  inline void AddElement(image_t image_id, const Eigen::Vector2d& image_xy);
  inline void AddElements(const std::vector<GCPTrackElement>& elements);

  // Delete existing element.
  inline void DeleteElement(size_t idx);
  void DeleteElement(image_t image_id, const Eigen::Vector2d& image_xy);

  // Requests that the track capacity be at least enough to contain the
  // specified number of elements.
  inline void Reserve(size_t num_elements);

  // Shrink the capacity of track vector to fit its size to save memory.
  inline void Compress();

 private:
  std::vector<GCPTrackElement> elements_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

size_t GCPTrack::Length() const { return elements_.size(); }

const std::vector<GCPTrackElement>& GCPTrack::Elements() const {
  return elements_;
}

std::vector<GCPTrackElement>& GCPTrack::Elements() { return elements_; }

void GCPTrack::SetElements(std::vector<GCPTrackElement> elements) {
  elements_ = std::move(elements);
}

// Access specific elements.
const GCPTrackElement& GCPTrack::Element(const size_t idx) const {
  return elements_.at(idx);
}

GCPTrackElement& GCPTrack::Element(const size_t idx) {
  return elements_.at(idx);
}

void GCPTrack::SetElement(const size_t idx, const GCPTrackElement& element) {
  elements_.at(idx) = element;
}

void GCPTrack::AddElement(const GCPTrackElement& element) {
  elements_.push_back(element);
}

void GCPTrack::AddElement(const image_t image_id,
                          const Eigen::Vector2d& image_xy) {
  elements_.emplace_back(image_id, image_xy);
}

void GCPTrack::AddElements(const std::vector<GCPTrackElement>& elements) {
  elements_.insert(elements_.end(), elements.begin(), elements.end());
}

void GCPTrack::DeleteElement(const size_t idx) {
  CHECK_LT(idx, elements_.size());
  elements_.erase(elements_.begin() + idx);
}

void GCPTrack::Reserve(const size_t num_elements) {
  elements_.reserve(num_elements);
}

void GCPTrack::Compress() { elements_.shrink_to_fit(); }

}  // namespace colmap
