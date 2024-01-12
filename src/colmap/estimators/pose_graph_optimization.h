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

#include "colmap/estimators/ceres_utils.h"
#include "colmap/scene/camera_rig.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/eigen_alignment.h"

#include <memory>
#include <unordered_set>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

struct PoseGraphOptimizationOptions {
  LossFunctionType loss_function_type = LossFunctionType::TRIVIAL;

  // Scaling factor determines residual at which robustification takes place.
  double loss_function_scale = 1.0;

  // Whether to refine the center parameter group.
  bool refine_center = true;

  // Whether to use prior rotation as constriant.
  bool use_prior_rotation = true;

  // Whether to use prior center as constriant.
  bool use_prior_center = true;

  // Whether to print a final summary.
  bool print_summary = true;

  // Minimum number of residuals to enable multi-threading. Note that
  // single-threaded is typically better for small bundle adjustment problems
  // due to the overhead of threading.
  int min_num_residuals_for_multi_threading = 50000;

  // Ceres-Solver options.
  ceres::Solver::Options solver_options;

  PoseGraphOptimizationOptions() {
    solver_options.function_tolerance = 0.0;
    solver_options.gradient_tolerance = 0.0;
    solver_options.parameter_tolerance = 0.0;
    solver_options.logging_type = ceres::LoggingType::SILENT;
    solver_options.max_num_iterations = 100;
    solver_options.max_linear_solver_iterations = 200;
    solver_options.max_num_consecutive_invalid_steps = 10;
    solver_options.max_consecutive_nonmonotonic_steps = 10;
    solver_options.num_threads = -1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = -1;
#endif  // CERES_VERSION_MAJOR
  }

  bool Check() const;
};

// Configuration container to setup pose graph optimization problems.
class PoseGraphOptimizationConfig {
 public:
  PoseGraphOptimizationConfig();

  size_t NumImages() const;
  size_t NumConstantCamPoses() const;
  size_t NumConstantCamPositionss() const;

  size_t NumImagePairs() const;
  size_t NumPriorPairs() const;

  // Determine the number of residuals for the given reconstruction. The number
  // of residuals equals the number of observations times two.
  size_t NumResiduals(const Reconstruction& reconstruction) const;

  // Add / remove images from the configuration.
  void AddImage(image_t image_id);
  bool HasImage(image_t image_id) const;
  void RemoveImage(image_t image_id);

  // Set the pose of added images as constant. The pose is defined as the
  // rotational and translational part of the projection matrix.
  void SetConstantCamPose(image_t image_id);
  void SetVariableCamPose(image_t image_id);
  bool HasConstantCamPose(image_t image_id) const;

  // Set the translational part of the pose, hence the constant pose
  // indices may be in [0, 1, 2] and must be unique. Note that the
  // corresponding images have to be added prior to calling these methods.
  void SetConstantCamPositions(image_t image_id, const std::vector<int>& idxs);
  void RemoveConstantCamPositions(image_t image_id);
  bool HasConstantCamPositions(image_t image_id) const;

  // Add / remove image pair from the configuration.
  void AddImagePair(image_pair_t pair_id);
  bool HasImagePair(image_pair_t pair_id);
  void RemoveImagePair(image_pair_t pair_id);

  // Add / remove prior pair from the configuration.
  void AddPriorPair(image_pair_t pair_id);
  bool HasPriorPair(image_pair_t pair_id);
  void RemovePriorPair(image_pair_t pair_id);

  // Access configuration data.
  const std::unordered_set<image_t>& Images() const;
  const std::vector<int>& ConstantCamPositions(image_t image_id) const;
  const std::unordered_set<image_pair_t>& ImagePairs() const;
  const std::unordered_set<image_pair_t>& PriorPairs() const;

 private:
  std::unordered_set<image_t> image_ids_;
  std::unordered_set<image_t> constant_cam_poses_;
  std::unordered_map<image_t, std::vector<int>> constant_cam_positions_;

  std::unordered_set<image_pair_t> image_pairs_;
  std::unordered_set<image_pair_t> prior_pairs_;
};

// Bundle adjustment based on Ceres-Solver. Enables most flexible configurations
// and provides best solution quality.
class PoseGraphOptimization {
 public:
  PoseGraphOptimization(const PoseGraphOptimizationOptions& options,
                        const PoseGraphOptimizationConfig& config);

  bool Solve(Reconstruction* reconstruction);

  // Get the Ceres solver summary for the last call to `Solve`.
  const ceres::Solver::Summary& Summary() const;

 private:
  void SetUp(Reconstruction* reconstruction,
             ceres::LossFunction* loss_function);
  void TearDown(Reconstruction* reconstruction);

  void AddPairToProblem(image_pair_t pair_id,
                        Reconstruction* reconstruction,
                        ceres::LossFunction* loss_function);

 protected:
  const PoseGraphOptimizationOptions options_;
  PoseGraphOptimizationConfig config_;
  std::unique_ptr<ceres::Problem> problem_;
  ceres::Solver::Summary summary_;
};

}  // namespace colmap
