#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// Eigen includes
#include <Eigen/Dense>
#include <Eigen/Core>

// xtensor includes  
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xbuilder.hpp>

using namespace std;
using namespace std::chrono;

// Benchmark timing utility
class Timer {
public:
    void start() { start_time = high_resolution_clock::now(); }
    double stop() {
        auto end_time = high_resolution_clock::now();
        return duration_cast<nanoseconds>(end_time - start_time).count() / 1e9;
    }
private:
    high_resolution_clock::time_point start_time;
};

// Prevent compiler from optimizing away computations
template<typename T>
void do_not_optimize(T&& value) {
    asm volatile("" : "+r,m"(value) : : "memory");
}

// Random number generator
mt19937 rng(42);
uniform_real_distribution<double> dist(-1.0, 1.0);

// Matrix-Vector Multiplication Benchmarks
namespace MatVecMul {
    
    // Serial C implementation
    void c_matvec(const vector<vector<double>>& A, const vector<double>& x, vector<double>& y) {
        size_t m = A.size();
        size_t n = A[0].size();
        for (size_t i = 0; i < m; ++i) {
            y[i] = 0.0;
            for (size_t j = 0; j < n; ++j) {
                y[i] += A[i][j] * x[j];
            }
        }
    }
    
    void benchmark(size_t m, size_t n, int iterations) {
        cout << "\nMatrix-Vector Multiplication (" << m << "x" << n << "):\n";
        
        Timer timer;
        
        // Initialize data
        vector<vector<double>> A_c(m, vector<double>(n));
        vector<double> x_c(n), y_c(m);
        
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A_c[i][j] = dist(rng);
            }
        }
        for (size_t i = 0; i < n; ++i) x_c[i] = dist(rng);
        
        // C benchmark
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            c_matvec(A_c, x_c, y_c);
            do_not_optimize(y_c[0]);  
        }
        double time_c = timer.stop();
        
        // Eigen benchmark
        Eigen::MatrixXd A_eigen(m, n);
        Eigen::VectorXd x_eigen(n), y_eigen(m);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A_eigen(i, j) = A_c[i][j];
            }
        }
        for (size_t i = 0; i < n; ++i) x_eigen(i) = x_c[i];
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            y_eigen = A_eigen * x_eigen;
            do_not_optimize(y_eigen[0]);  
        }
        double time_eigen = timer.stop();
        
        // xtensor benchmark - manual matrix-vector multiplication
        std::vector<size_t> shape_A = {m, n};
        std::vector<size_t> shape_x = {n};
        xt::xarray<double> A_xt = xt::zeros<double>(shape_A);
        xt::xarray<double> x_xt = xt::zeros<double>(shape_x);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                A_xt(i, j) = A_c[i][j];
            }
        }
        for (size_t i = 0; i < n; ++i) x_xt(i) = x_c[i];
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            std::vector<size_t> shape_y = {m};
            xt::xarray<double> y_xt = xt::zeros<double>(shape_y);
            for (size_t i = 0; i < m; ++i) {
                double sum = 0.0;
                for (size_t j = 0; j < n; ++j) {
                    sum += A_xt(i, j) * x_xt(j);
                }
                y_xt(i) = sum;
            }
            do_not_optimize(y_xt(0));  
        }
        double time_xt = timer.stop();
        
        cout << "  C:       " << fixed << setprecision(6) << time_c << "s\n";
        cout << "  Eigen:   " << fixed << setprecision(6) << time_eigen << "s\n"; 
        cout << "  xtensor: " << fixed << setprecision(6) << time_xt << "s\n";
    }
}

// Element-wise Operations
namespace ElementWise {
    
    void c_elementwise_exp(const vector<double>& x, vector<double>& y) {
        for (size_t i = 0; i < x.size(); ++i) {
            y[i] = exp(x[i] * x[i]) + sin(x[i]);
        }
    }
    
    void benchmark(size_t n, int iterations) {
        cout << "\nElement-wise Operations (exp(x²) + sin(x), n=" << n << "):\n";
        
        Timer timer;
        
        // Initialize data
        vector<double> x_c(n), y_c(n);
        for (size_t i = 0; i < n; ++i) x_c[i] = dist(rng);
        
        // C benchmark
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            c_elementwise_exp(x_c, y_c);
            do_not_optimize(y_c[0]);  
        }
        double time_c = timer.stop();
        
        // Eigen benchmark
        Eigen::VectorXd x_eigen(n);
        for (size_t i = 0; i < n; ++i) x_eigen(i) = x_c[i];
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            auto y_eigen = (x_eigen.array().square().exp() + x_eigen.array().sin()).matrix();
            do_not_optimize(y_eigen[0]);  
        }
        double time_eigen = timer.stop();
        
        // xtensor benchmark
        std::vector<size_t> shape_n = {n};
        xt::xarray<double> x_xt = xt::zeros<double>(shape_n);
        for (size_t i = 0; i < n; ++i) x_xt(i) = x_c[i];
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            auto y_xt = xt::exp(x_xt * x_xt) + xt::sin(x_xt);
            do_not_optimize(y_xt(0));  
        }
        double time_xt = timer.stop();
        
        cout << "  C:       " << fixed << setprecision(6) << time_c << "s\n";
        cout << "  Eigen:   " << fixed << setprecision(6) << time_eigen << "s\n";
        cout << "  xtensor: " << fixed << setprecision(6) << time_xt << "s\n";
    }
}

// Reduction Operations
namespace Reductions {
    
    double c_dot_product(const vector<double>& x, const vector<double>& y) {
        double result = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            result += x[i] * y[i];
        }
        return result;
    }
    
    void benchmark(size_t n, int iterations) {
        cout << "\nDot Product (n=" << n << "):\n";
        
        Timer timer;
        
        // Initialize data
        vector<double> x_c(n), y_c(n);
        for (size_t i = 0; i < n; ++i) {
            x_c[i] = dist(rng);
            y_c[i] = dist(rng);
        }
        
        // C benchmark
        double result_c;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            result_c = c_dot_product(x_c, y_c);
            do_not_optimize(result_c);  
        }
        double time_c = timer.stop();
        
        // Eigen benchmark
        Eigen::VectorXd x_eigen(n), y_eigen(n);
        for (size_t i = 0; i < n; ++i) {
            x_eigen(i) = x_c[i];
            y_eigen(i) = y_c[i];
        }
        
        double result_eigen;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            result_eigen = x_eigen.dot(y_eigen);
            do_not_optimize(result_eigen);  
        }
        double time_eigen = timer.stop();
        
        // xtensor benchmark
        xt::xarray<double> x_xt = xt::zeros<double>({n});
        xt::xarray<double> y_xt = xt::zeros<double>({n});
        for (size_t i = 0; i < n; ++i) {
            x_xt(i) = x_c[i];
            y_xt(i) = y_c[i];
        }
        
        double result_xt;
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            result_xt = xt::sum(x_xt * y_xt)();
            do_not_optimize(result_xt);  
        }
        double time_xt = timer.stop();
        
        cout << "  C:       " << fixed << setprecision(6) << time_c << "s\n";
        cout << "  Eigen:   " << fixed << setprecision(6) << time_eigen << "s\n";
        cout << "  xtensor: " << fixed << setprecision(6) << time_xt << "s\n";
    }
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define VCLIP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#define VFLOOR(x) floor(x)
#define VCUBIC(a, x) (((a[0] * (x) + a[1]) * (x) + a[2]) * (x) + a[3])

struct cubic_interp {
    double f, t0, length;
    double a[][4];
};

/*
 * Calculate coefficients of the interpolating polynomial in the form
 *      a[0] * t^3 + a[1] * t^2 + a[2] * t + a[3]
 */
static void cubic_interp_init_coefficients(
    double *a, const double *z, const double *z1)
{
    if (UNLIKELY(!isfinite(z1[1] + z1[2])))
    {
        /* If either of the inner grid points are NaN or infinite,
         * then fall back to nearest-neighbor interpolation. */
        a[0] = 0;
        a[1] = 0;
        a[2] = 0;
        a[3] = z[1];
    } else if (UNLIKELY(!isfinite(z1[0] + z1[3]))) {
        /* If either of the outer grid points are NaN or infinite,
         * then fall back to linear interpolation. */
        a[0] = 0;
        a[1] = 0;
        a[2] = z[2] - z[1];
        a[3] = z[1];
    } else {
        /* Otherwise, all of the grid points are finite.
         * Use cubic interpolation. */
        a[0] = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]);
        a[1] = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3];
        a[2] = 0.5 * (z[2] - z[0]);
        a[3] = z[1];
    }
}

cubic_interp *cubic_interp_init(
    const double *data, int n, double tmin, double dt)
{
    const int length = n + 6;
    cubic_interp *interp = (cubic_interp*)malloc(sizeof(*interp) + length * sizeof(*interp->a));
    if (LIKELY(interp))
    {
        interp->f = 1 / dt;
        interp->t0 = 3 - interp->f * tmin;
        interp->length = length;
        for (int i = 0; i < length; i ++)
        {
            double z[4];
            for (int j = 0; j < 4; j ++)
            {
                z[j] = data[VCLIP(i + j - 4, 0, n - 1)];
            }
            cubic_interp_init_coefficients(interp->a[i], z, z);
        }
    }
    return interp;
}

void cubic_interp_free(cubic_interp *interp)
{
    free(interp);
}

double cubic_interp_eval(const cubic_interp *interp, double t)
{
    if (UNLIKELY(isnan(t)))
        return t;
    double x = t, xmin = 0.0, xmax = interp->length - 1.0;
    x *= interp->f;
    x += interp->t0;
    x = VCLIP(x, xmin, xmax);
    double ix = VFLOOR(x);
    x -= ix;
    const double *a = interp->a[(int) ix];
    return VCUBIC(a, x);
}

namespace CubicInterpolation {
    
    void c_cubic_interpolate(const vector<double>& eval_points, 
                            cubic_interp* interp, 
                            vector<double>& results) {
        for (size_t i = 0; i < eval_points.size(); ++i) {
            results[i] = cubic_interp_eval(interp, eval_points[i]);
        }
    }
    
    // Eigen implementation
    void eigen_cubic_interpolate(const vector<double>& eval_points,
                                const Eigen::VectorXd& data,
                                double tmin, double dt,
                                Eigen::VectorXd& results) {
        double f = 1.0 / dt;
        double t0 = 3 - f * tmin;
        int n = data.size();
        
        for (size_t i = 0; i < eval_points.size(); ++i) {
            double t = eval_points[i];
            if (std::isnan(t)) {
                results[i] = t;
                continue;
            }
            
            double x = t * f + t0;
            x = std::max(0.0, std::min(x, (double)(n + 5)));
            int ix = (int)floor(x);
            x -= ix;
            
            // Get 4 data points for cubic interpolation
            Eigen::Vector4d z;
            for (int j = 0; j < 4; j++) {
                int idx = std::max(0, std::min(ix + j - 4, n - 1));
                z[j] = data[idx];
            }
            
            // Compute cubic coefficients
            Eigen::Vector4d a;
            if (!std::isfinite(z[1] + z[2])) {
                a << 0, 0, 0, z[1];
            } else if (!std::isfinite(z[0] + z[3])) {
                a << 0, 0, z[2] - z[1], z[1];
            } else {
                a[0] = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]);
                a[1] = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3];
                a[2] = 0.5 * (z[2] - z[0]);
                a[3] = z[1];
            }
            
            // Evaluate cubic polynomial
            results[i] = ((a[0] * x + a[1]) * x + a[2]) * x + a[3];
        }
    }
    
    // xtensor implementation
    void xtensor_cubic_interpolate(const vector<double>& eval_points,
                                  const xt::xarray<double>& data,
                                  double tmin, double dt,
                                  xt::xarray<double>& results) {
        double f = 1.0 / dt;
        double t0 = 3 - f * tmin;
        int n = data.size();
        
        for (size_t i = 0; i < eval_points.size(); ++i) {
            double t = eval_points[i];
            if (std::isnan(t)) {
                results[i] = t;
                continue;
            }
            
            double x = t * f + t0;
            x = std::max(0.0, std::min(x, (double)(n + 5)));
            int ix = (int)floor(x);
            x -= ix;
            
            // Get 4 data points for cubic interpolation
            std::vector<double> z(4);
            for (int j = 0; j < 4; j++) {
                int idx = std::max(0, std::min(ix + j - 4, n - 1));
                z[j] = data[idx];
            }
            
            // Compute cubic coefficients
            std::vector<double> a(4);
            if (!std::isfinite(z[1] + z[2])) {
                a[0] = 0; a[1] = 0; a[2] = 0; a[3] = z[1];
            } else if (!std::isfinite(z[0] + z[3])) {
                a[0] = 0; a[1] = 0; a[2] = z[2] - z[1]; a[3] = z[1];
            } else {
                a[0] = 1.5 * (z[1] - z[2]) + 0.5 * (z[3] - z[0]);
                a[1] = z[0] - 2.5 * z[1] + 2 * z[2] - 0.5 * z[3];
                a[2] = 0.5 * (z[2] - z[0]);
                a[3] = z[1];
            }
            
            // Evaluate cubic polynomial
            results[i] = ((a[0] * x + a[1]) * x + a[2]) * x + a[3];
        }
    }
    
    void benchmark(size_t n_data, size_t n_eval, int iterations) {
        cout << "\nCubic Interpolation (n_data=" << n_data << ", n_eval=" << n_eval << "):\n";
        
        Timer timer;
        
        // Create test data - a simple sinusoidal signal
        vector<double> data_c(n_data);
        double tmin = 0.0, tmax = 10.0;
        double dt = (tmax - tmin) / (n_data - 1);
        
        for (size_t i = 0; i < n_data; ++i) {
            double t = tmin + i * dt;
            data_c[i] = sin(2 * M_PI * t) + 0.5 * cos(4 * M_PI * t);
        }
        
        // Create evaluation points
        vector<double> eval_points(n_eval);
        for (size_t i = 0; i < n_eval; ++i) {
            eval_points[i] = tmin + (tmax - tmin) * dist(rng);
        }
        
        // C benchmark
        cubic_interp* interp_c = cubic_interp_init(data_c.data(), n_data, tmin, dt);
        vector<double> results_c(n_eval);
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            c_cubic_interpolate(eval_points, interp_c, results_c);
            do_not_optimize(results_c[0]);  
        }
        double time_c = timer.stop();
        
        cubic_interp_free(interp_c);
        
        // Eigen benchmark
        Eigen::VectorXd data_eigen(n_data);
        for (size_t i = 0; i < n_data; ++i) data_eigen[i] = data_c[i];
        Eigen::VectorXd results_eigen(n_eval);
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            eigen_cubic_interpolate(eval_points, data_eigen, tmin, dt, results_eigen);
            do_not_optimize(results_eigen[0]);  
        }
        double time_eigen = timer.stop();
        
        // xtensor benchmark
        std::vector<size_t> shape_data = {n_data};
        std::vector<size_t> shape_results = {n_eval};
        xt::xarray<double> data_xt = xt::zeros<double>(shape_data);
        xt::xarray<double> results_xt = xt::zeros<double>(shape_results);
        
        for (size_t i = 0; i < n_data; ++i) data_xt[i] = data_c[i];
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            xtensor_cubic_interpolate(eval_points, data_xt, tmin, dt, results_xt);
            do_not_optimize(results_xt[0]);  
        }
        double time_xt = timer.stop();
        
        cout << "  C:       " << fixed << setprecision(6) << time_c << "s\n";
        cout << "  Eigen:   " << fixed << setprecision(6) << time_eigen << "s\n";
        cout << "  xtensor: " << fixed << setprecision(6) << time_xt << "s\n";
        
        // Verify results are similar
        double max_diff = 0.0;
        for (size_t i = 0; i < std::min(n_eval, size_t(10)); ++i) {
            max_diff = std::max(max_diff, std::abs(results_c[i] - results_eigen[i]));
        }
        if (max_diff > 1e-10) {
            cout << "  Warning: Max difference between C and Eigen: " << max_diff << "\n";
        }
    }
}

// 3D Coordinate Transformations
namespace CoordinateTransforms {
    
    struct Vec3 { double x, y, z; };
    
    // Rotation matrix around Z-axis
    void c_rotation_z(const vector<Vec3>& points, vector<Vec3>& result, double angle) {
        double cos_a = cos(angle);
        double sin_a = sin(angle);
        
        for (size_t i = 0; i < points.size(); ++i) {
            result[i].x = cos_a * points[i].x - sin_a * points[i].y;
            result[i].y = sin_a * points[i].x + cos_a * points[i].y;
            result[i].z = points[i].z;
        }
    }
    
    void benchmark(size_t n_points, int iterations) {
        cout << "\n3D Rotation Transform (n=" << n_points << "):\n";
        
        Timer timer;
        double angle = 0.5;
        
        // Initialize data
        vector<Vec3> points_c(n_points), result_c(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            points_c[i] = {dist(rng), dist(rng), dist(rng)};
        }
        
        // C benchmark
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            c_rotation_z(points_c, result_c, angle);
            do_not_optimize(result_c[0].x);  
        }
        double time_c = timer.stop();
        
        // Eigen benchmark
        Eigen::Matrix3d R;
        R << cos(angle), -sin(angle), 0,
             sin(angle),  cos(angle), 0,
             0,           0,          1;
        
        Eigen::MatrixXd points_eigen(3, n_points);
        for (size_t i = 0; i < n_points; ++i) {
            points_eigen(0, i) = points_c[i].x;
            points_eigen(1, i) = points_c[i].y;
            points_eigen(2, i) = points_c[i].z;
        }
        
        timer.start();
        for (int i = 0; i < iterations; ++i) {
            auto result_eigen = R * points_eigen;
            do_not_optimize(result_eigen(0, 0));  
        }
        double time_eigen = timer.stop();
        
        // xtensor benchmark        
        std::vector<size_t> shape_R = {3, 3};
        std::vector<size_t> shape_points = {3, n_points};
        xt::xarray<double> R_xt = xt::zeros<double>(shape_R);
        xt::xarray<double> points_xt = xt::zeros<double>(shape_points);
        
        R_xt(0, 0) = cos(angle); R_xt(0, 1) = -sin(angle); R_xt(0, 2) = 0;
        R_xt(1, 0) = sin(angle); R_xt(1, 1) =  cos(angle); R_xt(1, 2) = 0;
        R_xt(2, 0) = 0;          R_xt(2, 1) = 0;           R_xt(2, 2) = 1;
        
        for (size_t i = 0; i < n_points; ++i) {
            points_xt(0, i) = points_c[i].x;
            points_xt(1, i) = points_c[i].y;
            points_xt(2, i) = points_c[i].z;
        }
        
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            xt::xarray<double> result_xt = xt::zeros<double>(shape_points);
            for (size_t i = 0; i < 3; ++i) {
                for (size_t j = 0; j < n_points; ++j) {
                    double sum = 0.0;
                    for (size_t k = 0; k < 3; ++k) {
                        sum += R_xt(i, k) * points_xt(k, j);
                    }
                    result_xt(i, j) = sum;
                }
            }
            do_not_optimize(result_xt(0, 0));  
        }
        double time_xt = timer.stop();
        
        cout << "  C:       " << fixed << setprecision(6) << time_c << "s\n";
        cout << "  Eigen:   " << fixed << setprecision(6) << time_eigen << "s\n";
        cout << "  xtensor: " << fixed << setprecision(6) << time_xt << "s\n";
    }
}

int main() {        
    // Matrix-vector multiplication tests
    MatVecMul::benchmark(100, 100, 10000);
    MatVecMul::benchmark(1000, 1000, 100);
    MatVecMul::benchmark(5000, 5000, 10);
    
    // Element-wise operations
    ElementWise::benchmark(100000, 1000);
    ElementWise::benchmark(1000000, 100);
    
    // Reduction operations
    Reductions::benchmark(100000, 1000);
    Reductions::benchmark(1000000, 100);
        
    // Cubic interpolation
    CubicInterpolation::benchmark(1000, 10000, 100); 
    CubicInterpolation::benchmark(10000, 50000, 10); 
    
    // 3D coordinate transformations
    CoordinateTransforms::benchmark(10000, 1000);
    CoordinateTransforms::benchmark(100000, 100);
        
    return 0;
}