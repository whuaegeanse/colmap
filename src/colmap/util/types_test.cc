// Copyright (c), ETH Zurich and UNC Chapel Hill.
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

#include "colmap/util/types.h"

#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace colmap {
namespace {

TEST(FilterView, Empty) {
  const std::vector<int> container;
  filter_view filtered_container(
      [](const int&) { return true; }, container.begin(), container.end());
  EXPECT_THAT(
      std::vector<int>(filtered_container.begin(), filtered_container.end()),
      testing::IsEmpty());
}

TEST(FilterView, All) {
  const std::vector<int> container = {1, 2, 3, 4, 5, 6};
  filter_view filtered_container(
      [](const int&) { return true; }, container.begin(), container.end());
  EXPECT_THAT(
      std::vector<int>(filtered_container.begin(), filtered_container.end()),
      container);
}

TEST(FilterView, None) {
  const std::vector<int> container = {1, 2, 3, 4, 5, 6};
  filter_view filtered_container(
      [](const int&) { return false; }, container.begin(), container.end());
  EXPECT_THAT(
      std::vector<int>(filtered_container.begin(), filtered_container.end()),
      testing::IsEmpty());
}

TEST(FilterView, Nominal) {
  const std::vector<int> container = {1, 2, 3, 4, 5, 6};
  filter_view filtered_container([](const int& d) { return d % 2 == 0; },
                                 container.begin(),
                                 container.end());
  EXPECT_THAT(
      std::vector<int>(filtered_container.begin(), filtered_container.end()),
      testing::ElementsAre(2, 4, 6));
}

TEST(FilterView, RangeExpression) {
  const std::vector<int> container = {1, 2, 3, 4, 5, 6};
  filter_view filtered_container([](const int& d) { return d % 2 == 0; },
                                 container.begin(),
                                 container.end());
  for (const int d : filtered_container) {
    EXPECT_EQ(d % 2, 0);
  }
}

TEST(FeatureMatchHashing, Nominal) {
  std::unordered_set<std::pair<point2D_t, point2D_t>> set;
  set.emplace(1, 2);
  EXPECT_EQ(set.size(), 1);
  set.emplace(1, 2);
  EXPECT_EQ(set.size(), 1);
  EXPECT_EQ(set.count(std::make_pair(0, 0)), 0);
  EXPECT_EQ(set.count(std::make_pair(1, 2)), 1);
  EXPECT_EQ(set.count(std::make_pair(2, 1)), 0);
  set.emplace(2, 1);
  EXPECT_EQ(set.size(), 2);
  EXPECT_EQ(set.count(std::make_pair(0, 0)), 0);
  EXPECT_EQ(set.count(std::make_pair(1, 2)), 1);
  EXPECT_EQ(set.count(std::make_pair(2, 1)), 1);
}

}  // namespace
}  // namespace colmap
