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

#include "colmap/estimators/pose_graph_optimization.h"

#include "colmap/estimators/cost_functions.h"
#include "colmap/scene/projection.h"
#include "colmap/sensor/models.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <iomanip>

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// PoseGraphOptimizationOptions
////////////////////////////////////////////////////////////////////////////////

bool PoseGraphOptimizationOptions::Check() const {
  CHECK_OPTION_GE(loss_function_scale, 0);
  return true;
}

////////////////////////////////////////////////////////////////////////////////
// PoseGraphOptimizationConfig
////////////////////////////////////////////////////////////////////////////////

PoseGraphOptimizationConfig::PoseGraphOptimizationConfig() {}

size_t PoseGraphOptimizationConfig::NumImages() const {
  return image_ids_.size();
}

size_t PoseGraphOptimizationConfig::NumConstantCamPoses() const {
  return constant_cam_poses_.size();
}

size_t PoseGraphOptimizationConfig::NumConstantCamPositionss() const {
  return constant_cam_positions_.size();
}

size_t PoseGraphOptimizationConfig::NumImagePairs() const {
  return image_pairs_.size();
}

size_t PoseGraphOptimizationConfig::NumPriorPairs() const {
  return prior_pairs_.size();
}

size_t PoseGraphOptimizationConfig::NumResiduals(
    const Reconstruction& reconstruction) const {
  // Count the number of observations for all added images.
  size_t num_observations = 0;
  for (const image_t image_id : image_ids_) {
    num_observations += reconstruction.Image(image_id).NumPoints3D();
  }

  // Count the number of observations for all added 3D points that are not
  // already added as part of the images above.

  auto NumObservationsForPoint = [this,
                                  &reconstruction](const point3D_t point3D_id) {
    size_t num_observations_for_point = 0;
    const auto& point3D = reconstruction.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (image_ids_.count(track_el.image_id) == 0) {
        num_observations_for_point += 1;
      }
    }
    return num_observations_for_point;
  };

  return 2 * num_observations;
}

void PoseGraphOptimizationConfig::AddImage(const image_t image_id) {
  image_ids_.insert(image_id);
}

bool PoseGraphOptimizationConfig::HasImage(const image_t image_id) const {
  return image_ids_.find(image_id) != image_ids_.end();
}

void PoseGraphOptimizationConfig::RemoveImage(const image_t image_id) {
  image_ids_.erase(image_id);
}

void PoseGraphOptimizationConfig::SetConstantCamPose(const image_t image_id) {
  CHECK(HasImage(image_id));
  CHECK(!HasConstantCamPositions(image_id));
  constant_cam_poses_.insert(image_id);
}

void PoseGraphOptimizationConfig::SetVariableCamPose(const image_t image_id) {
  constant_cam_poses_.erase(image_id);
}

bool PoseGraphOptimizationConfig::HasConstantCamPose(
    const image_t image_id) const {
  return constant_cam_poses_.find(image_id) != constant_cam_poses_.end();
}

void PoseGraphOptimizationConfig::SetConstantCamPositions(
    const image_t image_id, const std::vector<int>& idxs) {
  CHECK_GT(idxs.size(), 0);
  CHECK_LE(idxs.size(), 3);
  CHECK(HasImage(image_id));
  CHECK(!HasConstantCamPose(image_id));
  CHECK(!VectorContainsDuplicateValues(idxs))
      << "Tvec indices must not contain duplicates";
  constant_cam_positions_.emplace(image_id, idxs);
}

void PoseGraphOptimizationConfig::RemoveConstantCamPositions(
    const image_t image_id) {
  constant_cam_positions_.erase(image_id);
}

bool PoseGraphOptimizationConfig::HasConstantCamPositions(
    const image_t image_id) const {
  return constant_cam_positions_.find(image_id) !=
         constant_cam_positions_.end();
}

void PoseGraphOptimizationConfig::AddImagePair(image_pair_t pair_id) {
  image_pairs_.emplace(pair_id);

  image_t image_id1, image_id2;
  Database::PairIdToImagePair(pair_id, &image_id1, &image_id2);

  image_ids_.emplace(image_id1);
  image_ids_.emplace(image_id2);
}

bool PoseGraphOptimizationConfig::HasImagePair(image_pair_t pair_id) {
  return image_pairs_.find(pair_id) != image_pairs_.end();
}

void PoseGraphOptimizationConfig::RemoveImagePair(image_pair_t pair_id) {
  image_pairs_.erase(pair_id);
}

void PoseGraphOptimizationConfig::AddPriorPair(image_pair_t pair_id) {
  prior_pairs_.emplace(pair_id);
}

bool PoseGraphOptimizationConfig::HasPriorPair(image_pair_t pair_id) {
  return prior_pairs_.find(pair_id) != prior_pairs_.end();
}

void PoseGraphOptimizationConfig::RemovePriorPair(image_pair_t pair_id) {
  prior_pairs_.erase(pair_id);
}

const std::unordered_set<image_t>& PoseGraphOptimizationConfig::Images() const {
  return image_ids_;
}

const std::vector<int>& PoseGraphOptimizationConfig::ConstantCamPositions(
    const image_t image_id) const {
  return constant_cam_positions_.at(image_id);
}

const std::unordered_set<image_pair_t>&
PoseGraphOptimizationConfig::ImagePairs() const {
  return image_pairs_;
}

const std::unordered_set<image_pair_t>&
PoseGraphOptimizationConfig::PriorPairs() const {
  return prior_pairs_;
}

////////////////////////////////////////////////////////////////////////////////
// PoseGraphOptimization
////////////////////////////////////////////////////////////////////////////////

PoseGraphOptimization::PoseGraphOptimization(
    const PoseGraphOptimizationOptions& options,
    const PoseGraphOptimizationConfig& config)
    : options_(options), config_(config) {
  CHECK(options_.Check());
}

bool PoseGraphOptimization::Solve(Reconstruction* reconstruction) {
  CHECK_NOTNULL(reconstruction);
  CHECK(!problem_)
      << "Cannot use the same PoseGraphOptimization multiple times";

  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);

  std::unique_ptr<ceres::LossFunction> loss_function(CreateLossFunction(
      options_.loss_function_type, options_.loss_function_scale));
  SetUp(reconstruction, loss_function.get());

  if (problem_->NumResiduals() == 0) {
    return false;
  }

  ceres::Solver::Options solver_options = options_.solver_options;
  const bool has_sparse =
      solver_options.sparse_linear_algebra_library_type != ceres::NO_SPARSE;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 1000;
  const size_t num_images = config_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver && has_sparse) {
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::CGNR;
    solver_options.preconditioner_type = ceres::JACOBI;
  }

  if (problem_->NumResiduals() <
      options_.min_num_residuals_for_multi_threading) {
    solver_options.num_threads = 1;
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads = 1;
#endif  // CERES_VERSION_MAJOR
  } else {
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
    solver_options.num_linear_solver_threads =
        GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  }

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);

  if (options_.print_summary) {
    PrintHeading2("Pose graph optimization report");
    PrintSolverSummary(summary_);
  }

  TearDown(reconstruction);

  return true;
}

const ceres::Solver::Summary& PoseGraphOptimization::Summary() const {
  return summary_;
}

void PoseGraphOptimization::SetUp(Reconstruction* reconstruction,
                                  ceres::LossFunction* loss_function) {
  for (const image_t image_id : config_.Images()) {
    Image& image = reconstruction->Image(image_id);

    // CostFunction assumes unit quaternions.

    auto& cam_from_world = image.CamFromWorld();
    cam_from_world.rotation.normalize();

    if (options_.refine_center) {
      cam_from_world.UpdateCenter();
    }
  }

  // Warning: AddPointsToProblem assumes that AddImageToProblem is called first.
  // Do not change order of instructions!
  for (const image_pair_t pair_id : config_.ImagePairs()) {
    AddPairToProblem(pair_id, reconstruction, loss_function);
  }

  if (options_.use_prior_center) {
    for (image_t image_id : config_.Images()) {
      AddPriorToProblem(image_id, reconstruction, loss_function);
    }
  }
}

void PoseGraphOptimization::TearDown(Reconstruction* reconstruction) {
  for (const image_t image_id : config_.Images()) {
    Image& image = reconstruction->Image(image_id);

    // CostFunction assumes unit quaternions.

    auto& cam_from_world = image.CamFromWorld();
    cam_from_world.rotation.normalize();

    if (options_.refine_center) {
      cam_from_world.UpdateTranslation();
    }
  }
}

void PoseGraphOptimization::AddPairToProblem(
    image_pair_t pair_id,
    Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  image_t image_id1, image_id2;

  Database::PairIdToImagePair(pair_id, &image_id1, &image_id2);

  Image& image1 = reconstruction->Image(image_id1);
  Image& image2 = reconstruction->Image(image_id2);

  const bool constant_cam_pose1 = config_.HasConstantCamPose(image_id1);
  const bool constant_cam_pose2 = config_.HasConstantCamPose(image_id2);

  if (constant_cam_pose1 && constant_cam_pose2) {
    return;
  }

  if (options_.refine_center) {
    double* cam_from_world_rotation1 =
        image1.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_center1 = image1.CamFromWorld().center.data();

    double* cam_from_world_rotation2 =
        image2.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_center2 = image2.CamFromWorld().center.data();

    Eigen::Quaterniond q_ab_measured;
    Eigen::Vector3d p_ab_measured;
    double weight_rotation = 1.0;
    double weight_position = 1.0;

    // Add residuals to pose graph optimization problem.
    ceres::CostFunction* cost_function =
        center::PoseGraphErrorCostFunction::Create(
            q_ab_measured, p_ab_measured, weight_rotation, weight_position);

    problem_->AddResidualBlock(cost_function,
                               loss_function,
                               cam_from_world_rotation1,
                               cam_from_world_center1,
                               cam_from_world_rotation2,
                               cam_from_world_center2);

    // Set pose parameterization.
    if (!constant_cam_pose1) {
      SetQuaternionManifold(problem_.get(), cam_from_world_rotation1);
      const std::vector<int>& constant_position_idxs =
          config_.ConstantCamPositions(image_id1);
      SetSubsetManifold(
          3, constant_position_idxs, problem_.get(), cam_from_world_center1);
    }

    if (!constant_cam_pose2) {
      SetQuaternionManifold(problem_.get(), cam_from_world_rotation1);
      const std::vector<int>& constant_position_idxs =
          config_.ConstantCamPositions(image_id2);
      SetSubsetManifold(
          3, constant_position_idxs, problem_.get(), cam_from_world_center2);
    }
  } else {
    double* cam_from_world_rotation1 =
        image1.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_translation1 =
        image1.CamFromWorld().translation.data();

    double* cam_from_world_rotation2 =
        image2.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_translation2 =
        image2.CamFromWorld().translation.data();

    Eigen::Quaterniond q_ab_measured;
    Eigen::Vector3d p_ab_measured;
    double weight_rotation = 1.0;
    double weight_position = 1.0;

    // Add residuals to pose graph optimization problem.
    ceres::CostFunction* cost_function =
        translantion::PoseGraphErrorCostFunction::Create(
            q_ab_measured, p_ab_measured, weight_rotation, weight_position);

    problem_->AddResidualBlock(cost_function,
                               loss_function,
                               cam_from_world_rotation1,
                               cam_from_world_translation1,
                               cam_from_world_rotation2,
                               cam_from_world_translation2);

    // Set pose parameterization.
    if (!constant_cam_pose1) {
      SetQuaternionManifold(problem_.get(), cam_from_world_rotation1);
      const auto& constant_position_idxs =
          config_.ConstantCamPositions(image_id1);
      SetSubsetManifold(3,
                        constant_position_idxs,
                        problem_.get(),
                        cam_from_world_translation1);
    }

    if (!constant_cam_pose2) {
      SetQuaternionManifold(problem_.get(), cam_from_world_rotation1);
      const auto& constant_position_idxs =
          config_.ConstantCamPositions(image_id2);
      SetSubsetManifold(3,
                        constant_position_idxs,
                        problem_.get(),
                        cam_from_world_translation2);
    }
  }
}

void PoseGraphOptimization::AddPriorToProblem(
    image_t image_id,
    Reconstruction* reconstruction,
    ceres::LossFunction* loss_function) {
  Image& image = reconstruction->Image(image_id);

  const bool constant_cam_pose = config_.HasConstantCamPose(image_id);

  if (constant_cam_pose) {
    return;
  }

  if (options_.refine_center) {
    double* cam_from_world_rotation =
        image.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_center = image.CamFromWorld().center.data();

    Eigen::Vector3d p_measured;
    double weight = 1.0;

    // Add residuals to pose graph optimization problem.
    ceres::CostFunction* cost_function =
        center::GNSSErrorCostFunction::Create(p_measured, weight);

    problem_->AddResidualBlock(cost_function,
                               loss_function,
                               cam_from_world_rotation,
                               cam_from_world_center);

    // Set pose parameterization.
    if (!constant_cam_pose) {
      const auto& constant_position_idxs =
          config_.ConstantCamPositions(image_id);
      SetSubsetManifold(
          3, constant_position_idxs, problem_.get(), cam_from_world_center);
    }
  } else {
    double* cam_from_world_rotation =
        image.CamFromWorld().rotation.coeffs().data();
    double* cam_from_world_translation =
        image.CamFromWorld().translation.data();

    Eigen::Vector3d p_measured;
    double weight = 1.0;

    // Add residuals to pose graph optimization problem.
    ceres::CostFunction* cost_function =
        translantion::GNSSErrorCostFunction::Create(p_measured, weight);

    problem_->AddResidualBlock(cost_function,
                               loss_function,
                               cam_from_world_rotation,
                               cam_from_world_translation);

    // Set pose parameterization.
    if (!constant_cam_pose) {
      const auto& constant_position_idxs =
          config_.ConstantCamPositions(image_id);
      SetSubsetManifold(3,
                        constant_position_idxs,
                        problem_.get(),
                        cam_from_world_translation);
    }
  }
}

}  // namespace colmap
