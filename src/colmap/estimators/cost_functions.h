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

#include "colmap/geometry/rigid3.h"
#include "colmap/sensor/models.h"
#include "colmap/util/eigen_alignment.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

template <typename T>
using EigenVector3Map = Eigen::Map<const Eigen::Matrix<T, 3, 1>>;
template <typename T>
using EigenQuaternionMap = Eigen::Map<const Eigen::Quaternion<T>>;
using EigenMatrix6d = Eigen::Matrix<double, 6, 6>;

// Standard bundle adjustment cost function for variable
// camera pose, calibration, and point parameters.
template <typename CameraModel>
class ReprojErrorCostFunction {
 public:
  explicit ReprojErrorCostFunction(const Eigen::Vector2d& point2D,
                                   const double weight = 1.0)
      : observed_x_(point2D(0)), observed_y_(point2D(1)), weight_(weight) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                     const double weight = 1.0) {
    return (
        new ceres::AutoDiffCostFunction<ReprojErrorCostFunction<CameraModel>,
                                        2,
                                        4,
                                        3,
                                        3,
                                        CameraModel::num_params>(
            new ReprojErrorCostFunction(point2D, weight)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world_rotation) *
            EigenVector3Map<T>(point3D) +
        EigenVector3Map<T>(cam_from_world_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] = T(weight_) * (residuals[0] - T(observed_x_));
    residuals[1] = T(weight_) * (residuals[1] - T(observed_y_));
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
  const double weight_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class ReprojErrorConstantPoseCostFunction {
 public:
  ReprojErrorConstantPoseCostFunction(const Rigid3d& cam_from_world,
                                      const Eigen::Vector2d& point2D,
                                      const double weight = 1.0)
      : cam_from_world_(cam_from_world),
        observed_x_(point2D(0)),
        observed_y_(point2D(1)),
        weight_(weight) {}

  static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                     const Eigen::Vector2d& point2D,
                                     const double weight = 1.0) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPoseCostFunction<CameraModel>,
            2,
            3,
            CameraModel::num_params>(new ReprojErrorConstantPoseCostFunction(
        cam_from_world, point2D, weight)));
  }

  template <typename T>
  bool operator()(const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        cam_from_world_.rotation.cast<T>() * EigenVector3Map<T>(point3D) +
        cam_from_world_.translation.cast<T>();
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] = T(weight_) * (residuals[0] - T(observed_x_));
    residuals[1] = T(weight_) * (residuals[1] - T(observed_y_));
    return true;
  }

 private:
  const Rigid3d& cam_from_world_;
  const double observed_x_;
  const double observed_y_;
  const double weight_;
};

// Bundle adjustment cost function for variable
// camera pose and calibration parameters, and fixed point.
template <typename CameraModel>
class ReprojErrorConstantPoint3DCostFunction {
 public:
  ReprojErrorConstantPoint3DCostFunction(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const double weight = 1.0)
      : observed_x_(point2D(0)),
        observed_y_(point2D(1)),
        point3D_x_(point3D(0)),
        point3D_y_(point3D(1)),
        point3D_z_(point3D(2)),
        weight_(weight) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                     const Eigen::Vector3d& point3D,
                                     const double weight = 1.0) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPoint3DCostFunction<CameraModel>,
            2,
            4,
            3,
            CameraModel::num_params>(
        new ReprojErrorConstantPoint3DCostFunction(point2D, point3D, weight)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  const T* const camera_params,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 1> point3D;
    point3D[0] = T(point3D_x_);
    point3D[1] = T(point3D_y_);
    point3D[2] = T(point3D_z_);

    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_world_rotation) * point3D +
        EigenVector3Map<T>(cam_from_world_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] = T(weight_) * (residuals[0] - T(observed_x_));
    residuals[1] = T(weight_) * (residuals[1] - T(observed_y_));
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
  const double point3D_x_;
  const double point3D_y_;
  const double point3D_z_;
  const double weight_;
};

// Bundle adjustment cost function for variable
// camera pose and calibration parameters, and fixed point.
template <typename CameraModel>
class ReprojErrorConstantPosePoint3DCostFunction {
 public:
  ReprojErrorConstantPosePoint3DCostFunction(const Rigid3d& cam_from_world,
                                             const Eigen::Vector2d& point2D,
                                             const Eigen::Vector3d& point3D,
                                             const double weight = 1.0)
      : cam_from_world_(cam_from_world),
        observed_x_(point2D(0)),
        observed_y_(point2D(1)),
        point3D_x_(point3D(0)),
        point3D_y_(point3D(1)),
        point3D_z_(point3D(2)),
        weight_(weight) {}

  static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                     const Eigen::Vector2d& point2D,
                                     const Eigen::Vector3d& point3D,
                                     const double weight = 1.0) {
    return (new ceres::AutoDiffCostFunction<
            ReprojErrorConstantPosePoint3DCostFunction<CameraModel>,
            2,
            CameraModel::num_params>(
        new ReprojErrorConstantPosePoint3DCostFunction(
            cam_from_world, point2D, point3D, weight)));
  }

  template <typename T>
  bool operator()(const T* const camera_params, T* residuals) const {
    Eigen::Matrix<T, 3, 1> point3D;
    point3D[0] = T(point3D_x_);
    point3D[1] = T(point3D_y_);
    point3D[2] = T(point3D_z_);

    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        cam_from_world_.rotation.cast<T>() * point3D +
        cam_from_world_.translation.cast<T>();

    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] = T(weight_) * (residuals[0] - T(observed_x_));
    residuals[1] = T(weight_) * (residuals[1] - T(observed_y_));
    return true;
  }

 private:
  const Rigid3d& cam_from_world_;
  const double observed_x_;
  const double observed_y_;
  const double point3D_x_;
  const double point3D_y_;
  const double point3D_z_;
  const double weight_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigReprojErrorCostFunction {
 public:
  explicit RigReprojErrorCostFunction(const Eigen::Vector2d& point2D,
                                      const double weight = 1.0)
      : observed_x_(point2D(0)), observed_y_(point2D(1)), weight_(weight) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D,
                                     const double weight = 1.0) {
    return (
        new ceres::AutoDiffCostFunction<RigReprojErrorCostFunction<CameraModel>,
                                        2,
                                        4,
                                        3,
                                        4,
                                        3,
                                        3,
                                        CameraModel::num_params>(
            new RigReprojErrorCostFunction(point2D, weight)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_rig_rotation,
                  const T* const cam_from_rig_translation,
                  const T* const rig_from_world_rotation,
                  const T* const rig_from_world_translation,
                  const T* const point3D,
                  const T* const camera_params,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> point3D_in_cam =
        EigenQuaternionMap<T>(cam_from_rig_rotation) *
            (EigenQuaternionMap<T>(rig_from_world_rotation) *
                 EigenVector3Map<T>(point3D) +
             EigenVector3Map<T>(rig_from_world_translation)) +
        EigenVector3Map<T>(cam_from_rig_translation);
    CameraModel::ImgFromCam(camera_params,
                            point3D_in_cam[0],
                            point3D_in_cam[1],
                            point3D_in_cam[2],
                            &residuals[0],
                            &residuals[1]);
    residuals[0] = T(weight_) * (residuals[0] - T(observed_x_));
    residuals[1] = T(weight_) * (residuals[1] - T(observed_y_));
    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
  const double weight_;
};

// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement. We have two poses x_a
// and x_b. Through sensor measurements we can measure the transformation of
// frame B w.r.t frame A denoted as t_ab_hat. We can compute an error metric
// between the current estimate of the poses and the measurement.
//
// In this formulation, we have chosen to represent the rigid transformation as
// a Hamiltonian quaternion, q, and position, p. The quaternion ordering is
// [x, y, z, w].

// The estimated measurement is:
//      t_ab = [ p_ab ]  = [ R(q_a)^T * (p_b - p_a) ]
//             [ q_ab ]    [ q_a^{-1] * q_b         ]
//
// where ^{-1} denotes the inverse and R(q) is the rotation matrix for the
// quaternion. Now we can compute an error metric between the estimated and
// measurement transformation. For the orientation error, we will use the
// standard multiplicative error resulting in:
//
//   error = [ p_ab - \hat{p}_ab                 ]
//           [ 2.0 * Vec(q_ab * \hat{q}_ab^{-1}) ]
//
// where Vec(*) returns the vector (imaginary) part of the quaternion. Since
// the measurement has an uncertainty associated with how accurate it is, we
// will weight the errors by the square root of the measurement information
// matrix:
//
//   residuals = I^{1/2) * error
// where I is the information matrix which is the inverse of the covariance.
//
//
class PoseErrorCostFunction {
 public:
  PoseErrorCostFunction(const Eigen::Quaterniond& q_ab_measured,
                        const Eigen::Vector3d& p_ab_measured,
                        const double weight_rotation,
                        const double weight_position)
      : q_ab_measured_(q_ab_measured),
        p_ab_measured_(p_ab_measured),
        weight_rotation_(2.0 * weight_rotation),
        weight_position_(weight_position) {}

  static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ab_measured,
                                     const Eigen::Vector3d& p_ab_measured,
                                     const double weight_rotation = 1.0,
                                     const double weight_position = 1.0) {
    return (
        new ceres::AutoDiffCostFunction<PoseErrorCostFunction, 6, 4, 3, 4, 3>(
            new PoseErrorCostFunction(q_ab_measured,
                                      p_ab_measured,
                                      weight_rotation,
                                      weight_position)));
  }

  template <typename T>
  bool operator()(const T* const cam1_from_world_rotation,
                  const T* const cam1_from_world_translation,
                  const T* const cam2_from_world_rotation,
                  const T* const cam2_from_world_translation,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_a(cam1_from_world_translation);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(cam1_from_world_rotation);
    Eigen::Matrix<T, 3, 1> p_a = q_a * (-t_a);

    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_b(cam2_from_world_translation);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(cam2_from_world_rotation);
    Eigen::Matrix<T, 3, 1> p_b = q_b * (-t_b);

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        q_ab_measured_.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals_ref(residuals);
    residuals_ref.template block<3, 1>(3, 0) =
        T(weight_rotation_) * delta_q.vec();

    residuals_ref.template block<3, 1>(0, 0) =
        T(weight_position_) *
        (p_ab_estimated - p_ab_measured_.template cast<T>());

    return true;
  }

 private:
  Eigen::Quaterniond q_ab_measured_;
  Eigen::Vector3d p_ab_measured_;
  double weight_rotation_;
  double weight_position_;
};

class PoseCenterErrorCostFunction {
 public:
  PoseCenterErrorCostFunction(const Eigen::Vector3d& c_measured,
                              const double weight_x,
                              const double weight_y,
                              const double weight_z)
      : c_measured_(c_measured),
        weight_x_(weight_x),
        weight_y_(weight_y),
        weight_z_(weight_z) {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& p_measured,
                                     const double weight_x,
                                     const double weight_y,
                                     const double weight_z) {
    return (
        new ceres::AutoDiffCostFunction<PoseCenterErrorCostFunction, 3, 4, 3>(
            new PoseCenterErrorCostFunction(
                p_measured, weight_x, weight_y, weight_z)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_rotation,
                  const T* const cam_from_world_translation,
                  T* residuals) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_a(cam_from_world_translation);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(cam_from_world_rotation);

    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Matrix<T, 3, 1> c_estimated = q_a_inverse * (-t_a);

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_ref(residuals);
    residuals_ref = c_estimated - c_measured_.template cast<T>();

    residuals_ref(0) *= T(weight_x_);
    residuals_ref(1) *= T(weight_y_);
    residuals_ref(2) *= T(weight_z_);

    return true;
  }

 private:
  Eigen::Vector3d c_measured_;
  double weight_x_;
  double weight_y_;
  double weight_z_;
};

// Bundle adjustment cost function for variable point3d.
class Point3DErrorCostFunction {
 public:
  Point3DErrorCostFunction(const Eigen::Vector3d& p_measured,
                           const double weight_xy,
                           const double weight_z)
      : p_measured_x_(p_measured(0)),
        p_measured_y_(p_measured(1)),
        p_measured_z_(p_measured(2)),
        weight_xy_(weight_xy),
        weight_z_(weight_z) {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& p_measured,
                                     const double weight_xy,
                                     const double weight_z) {
    return (new ceres::AutoDiffCostFunction<Point3DErrorCostFunction, 3, 3>(
        new Point3DErrorCostFunction(p_measured, weight_xy, weight_z)));
  }

  template <typename T>
  bool operator()(const T* const p_estimated, T* residuals) const {
    residuals[0] = T(weight_xy_) * (p_estimated[0] - T(p_measured_x_));
    residuals[1] = T(weight_xy_) * (p_estimated[1] - T(p_measured_y_));
    residuals[2] = T(weight_z_) * (p_estimated[2] - T(p_measured_z_));
    return true;
  }

 private:
  const double p_measured_x_;
  const double p_measured_y_;
  const double p_measured_z_;

  const double weight_xy_;
  const double weight_z_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `SphereManifold`.
class SampsonErrorCostFunction {
 public:
  SampsonErrorCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<SampsonErrorCostFunction, 1, 4, 3>(
        new SampsonErrorCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const cam2_from_cam1_rotation,
                  const T* const cam2_from_cam1_translation,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 3> R =
        EigenQuaternionMap<T>(cam2_from_cam1_rotation).toRotationMatrix();

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -cam2_from_cam1_translation[2], cam2_from_cam1_translation[1],
        cam2_from_cam1_translation[2], T(0), -cam2_from_cam1_translation[0],
        -cam2_from_cam1_translation[1], cam2_from_cam1_translation[0], T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 /
                   (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                    Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const double x1_;
  const double y1_;
  const double x2_;
  const double y2_;
};

template <typename T>
inline void EigenQuaternionToAngleAxis(const T* eigen_quaternion,
                                       T* angle_axis) {
  const T quaternion[4] = {eigen_quaternion[3],
                           eigen_quaternion[0],
                           eigen_quaternion[1],
                           eigen_quaternion[2]};
  ceres::QuaternionToAngleAxis(quaternion, angle_axis);
}

// 6-DoF error on the absolute pose. The residual is the log of the error pose,
// splitting SE(3) into SO(3) x R^3. The 6x6 covariance matrix is defined in the
// reference frame of the camera. Its first and last three components correspond
// to the rotation and translation errors, respectively.
struct AbsolutePoseErrorCostFunction {
 public:
  AbsolutePoseErrorCostFunction(const Rigid3d& cam_from_world,
                                const EigenMatrix6d& covariance_cam)
      : world_from_cam_(Inverse(cam_from_world)),
        sqrt_information_cam_(covariance_cam.inverse().llt().matrixL()) {}

  static ceres::CostFunction* Create(const Rigid3d& cam_from_world,
                                     const EigenMatrix6d& covariance_cam) {
    return (
        new ceres::AutoDiffCostFunction<AbsolutePoseErrorCostFunction, 6, 4, 3>(
            new AbsolutePoseErrorCostFunction(cam_from_world, covariance_cam)));
  }

  template <typename T>
  bool operator()(const T* const cam_from_world_q,
                  const T* const cam_from_world_t,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> param_from_measured_q =
        EigenQuaternionMap<T>(cam_from_world_q) *
        world_from_cam_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals_ptr);

    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_t(residuals_ptr + 3);
    param_from_measured_t = EigenVector3Map<T>(cam_from_world_t) +
                            EigenQuaternionMap<T>(cam_from_world_q) *
                                world_from_cam_.translation.cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(sqrt_information_cam_.template cast<T>());
    return true;
  }

 private:
  const Rigid3d world_from_cam_;
  const EigenMatrix6d sqrt_information_cam_;
};

// 6-DoF error between two absolute poses based on a measurement that is their
// relative pose, with identical scale for the translation. The covariance is
// defined in the reference frame of the camera j.
// Its first and last three components correspond to the rotation and
// translation errors, respectively.
//
// Derivation:
// j_T_w = ΔT_j·j_T_i·i_T_w
// where ΔT_j = exp(η_j) is the residual in SE(3) and η_j in tangent space.
// Thus η_j = log(j_T_w·i_T_w⁻¹·i_T_j)
// Rotation term: ΔR = log(j_R_w·i_R_w⁻¹·i_R_j)
// Translation term: Δt = j_t_w + j_R_w·i_R_w⁻¹·(i_t_j -i_t_w)
struct MetricRelativePoseErrorCostFunction {
 public:
  MetricRelativePoseErrorCostFunction(const Rigid3d& i_from_j,
                                      const EigenMatrix6d& covariance_j)
      : i_from_j_(i_from_j),
        sqrt_information_j_(covariance_j.inverse().llt().matrixL()) {}

  static ceres::CostFunction* Create(const Rigid3d& i_from_j,
                                     const EigenMatrix6d& covariance_j) {
    return (new ceres::AutoDiffCostFunction<MetricRelativePoseErrorCostFunction,
                                            6,
                                            4,
                                            3,
                                            4,
                                            3>(
        new MetricRelativePoseErrorCostFunction(i_from_j, covariance_j)));
  }

  template <typename T>
  bool operator()(const T* const i_from_world_q,
                  const T* const i_from_world_t,
                  const T* const j_from_world_q,
                  const T* const j_from_world_t,
                  T* residuals_ptr) const {
    const Eigen::Quaternion<T> j_from_i_q =
        EigenQuaternionMap<T>(j_from_world_q) *
        EigenQuaternionMap<T>(i_from_world_q).inverse();
    const Eigen::Quaternion<T> param_from_measured_q =
        j_from_i_q * i_from_j_.rotation.cast<T>();
    EigenQuaternionToAngleAxis(param_from_measured_q.coeffs().data(),
                               residuals_ptr);

    Eigen::Matrix<T, 3, 1> i_from_jw_t =
        i_from_j_.translation.cast<T>() - EigenVector3Map<T>(i_from_world_t);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> param_from_measured_t(residuals_ptr + 3);
    param_from_measured_t =
        EigenVector3Map<T>(j_from_world_t) + j_from_i_q * i_from_jw_t;

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.applyOnTheLeft(sqrt_information_j_.template cast<T>());
    return true;
  }

 private:
  const Rigid3d& i_from_j_;
  const EigenMatrix6d sqrt_information_j_;
};

template <template <typename> class CostFunction, typename... Args>
ceres::CostFunction* CameraCostFunction(const CameraModelId camera_model_id,
                                        Args&&... args) {
  switch (camera_model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                     \
  case CameraModel::model_id:                                              \
    return CostFunction<CameraModel>::Create(std::forward<Args>(args)...); \
    break;

    CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
  }
}

inline void SetQuaternionManifold(ceres::Problem* problem, double* quat_xyzw) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(quat_xyzw, new ceres::EigenQuaternionManifold);
#else
  problem->SetParameterization(quat_xyzw,
                               new ceres::EigenQuaternionParameterization);
#endif
}

inline void SetSubsetManifold(int size,
                              const std::vector<int>& constant_params,
                              ceres::Problem* problem,
                              double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params,
                       new ceres::SubsetManifold(size, constant_params));
#else
  problem->SetParameterization(
      params, new ceres::SubsetParameterization(size, constant_params));
#endif
}

template <int size>
inline void SetSphereManifold(ceres::Problem* problem, double* params) {
#if CERES_VERSION_MAJOR >= 3 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 1)
  problem->SetManifold(params, new ceres::SphereManifold<size>);
#else
  problem->SetParameterization(
      params, new ceres::HomogeneousVectorParameterization(size));
#endif
}

}  // namespace colmap
