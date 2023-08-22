#include <iostream>
#include <string>
#include <cmath>
#include <memory>
#include <algorithm>
#include <thread>

// V3
template <typename T>
struct V3
{
  T x, y, z;

  constexpr V3() noexcept
    : x(T{}), y(T{}), z(T{}) {}
  constexpr explicit V3(T t) noexcept
    : x(t), y(t), z(t) {}
  constexpr explicit V3(T x, T y, T z) noexcept
    : x(x), y(y), z(z) {}
};

template <typename T>
constexpr auto operator+(const V3<T>& lhs, const V3<T>& rhs) noexcept
{
  return V3<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}
template <typename T>
constexpr auto operator-(const V3<T>& lhs, const V3<T>& rhs) noexcept
{
  return V3<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}
template <typename T>
constexpr auto operator*(const V3<T>& lhs, T rhs) noexcept
{
  return V3<T>(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}
template <typename T>
constexpr auto operator*(T lhs, const V3<T>& rhs) noexcept
{
  return rhs * lhs;
}
template <typename T>
constexpr auto operator/(const V3<T>& lhs, T rhs) noexcept
{
  return V3<T>(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

template <typename T>
constexpr auto& operator+=(V3<T>& lhs, const V3<T>& rhs) noexcept
{
  lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z;
  return lhs;
}
template <typename T>
constexpr auto& operator-=(V3<T>& lhs, const V3<T>& rhs) noexcept
{
  lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z;
  return lhs;
}
template <typename T>
constexpr auto& operator*=(V3<T>& lhs, T rhs) noexcept
{
  lhs.x *= rhs; lhs.y *= rhs; lhs.z *= rhs;
  return lhs;
}
template <typename T>
constexpr auto& operator/=(V3<T>& lhs, T rhs) noexcept
{
  lhs.x /= rhs; lhs.y /= rhs; lhs.z /= rhs;
  return lhs;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const V3<T>& v)
{
  os << "[" << v.x << ", " << v.y << ", " << v.z << "]\n"; 
  return os;
}

template <typename T>
constexpr auto dot(const V3<T>& lhs, const V3<T>& rhs) noexcept
{
  return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}
template <typename T>
constexpr auto cross(const V3<T>& lhs, const V3<T>& rhs) noexcept
{
  return V3<T>(
    lhs.y*rhs.z - lhs.z*rhs.y,
    lhs.z*rhs.x - lhs.x*rhs.z,
    lhs.x*rhs.y - lhs.y*rhs.x
  );
}
template <typename T>
auto length(const V3<T>& v) noexcept
{
  return std::sqrt(dot(v, v));
}
template <typename T>
auto normalize(const V3<T>& v) noexcept
{
  return v / length(v);
}
template <typename T>
auto distance(const V3<T>& lhs, const V3<T>& rhs) noexcept
{
  return length(lhs - rhs);
}
template <typename T>
constexpr auto element_wise_max(const V3<T>& lhs, T rhs)
{
  return V3<T>(
    std::max(lhs.x, rhs),
    std::max(lhs.y, rhs),
    std::max(lhs.z, rhs)
  );
}
template <typename T>
auto element_wise_abs(const V3<T>& v)
{
  return V3<T>(
    std::abs(v.x),
    std::abs(v.y),
    std::abs(v.z)
  );
}
template <typename T>
auto rotate(const V3<T>& input, const V3<T>& rotation_unit, T angle)
{
  const auto c = std::cos(angle);
  return
    input * c
    + cross(rotation_unit, input) * std::sin(angle)
    + rotation_unit * dot(rotation_unit, input) * (T{1} - c);
}

using V3d = V3<double>;
// V3 end

class RayMarcher
{
private:

  using Clock = std::chrono::steady_clock;

  const std::size_t xr_, yr_;
  std::string       pixels_;
  Clock::time_point time_origin_;
  double            time_;

  static char get_char(double t)
  {
    static const std::string table = " .:-=+*#%@";

    auto index = static_cast<std::size_t>(std::floor(
      std::clamp(t, 0.0, 1.0) * static_cast<double>(table.size())
    ));
    index = std::min(index, table.size() - 1);

    return table[index];
  }

  static double box(const V3d& p, const V3d& b)
  {
    const auto q = element_wise_abs(p) - b;
    return length(element_wise_max(q, 0.0)) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.0);
  }
  
  double scene(const V3d& p)
  {
    static constexpr auto box_dim = V3d(0.15);
    return box(rotate(p, normalize(V3d(3., 7., 5.)), time_), box_dim);
  }

  V3d normal(const V3d& p, double eps)
  {
    return normalize(V3d(
      scene(V3d(p.x + eps, p.y, p.z)) - scene(V3d(p.x - eps, p.y, p.z)),
      scene(V3d(p.x, p.y + eps, p.z)) - scene(V3d(p.x, p.y - eps, p.z)),
      scene(V3d(p.x, p.y, p.z + eps)) - scene(V3d(p.x, p.y, p.z - eps))
    ));
  }

  char calculate(std::size_t x, std::size_t y)
  {
    // pixels are rectangular
    static constexpr double      x_fix     = 0.425;
    static constexpr double      eps       = 1.0E-6;
    static constexpr double      give_up   = 1.0E+6;
    static constexpr std::size_t max_steps = 64;
    static constexpr auto        light_pos = V3d(3., 3., -3.);
    static constexpr double      light_amp = 200.;
    static const     auto        light_dir = normalize(V3d(1.0, 1.0, -1.0));

    const auto xrd   = static_cast<double>(xr_) * x_fix;
    const auto yrd   = static_cast<double>(yr_);
    const auto xd    = static_cast<double>(x) * x_fix;
    const auto yd    = static_cast<double>(y);
    const auto r_max = static_cast<double>(std::max(xr_, yr_));

    const auto uv    = 2.0 * (V3d(xd, yd, 0.0) - 0.5 * V3d(xrd, yrd, 0.0)) / r_max;
    const auto eye   = V3d(0.5, 0.5, -1.0);
    const auto ray   = normalize(uv - eye);

    bool   hit   = false;
    double depth = 0.0;

    for (std::size_t i = 0; i < max_steps; ++i) {
      const auto pos = eye + depth * ray;

      double dist = scene(pos);
      if (dist < eps) {
        hit = true;
        break;
      }

      depth += dist;

      if (dist > give_up) {
        break;
      }
    }

    if (hit) {
      const auto pos = eye + depth * ray;
      return get_char(
        light_amp * dot(normal(pos, eps), light_dir) * std::exp(-distance(pos, light_pos))
      );
    }
    else {
      return get_char(0.);
    }
  }

  void reset_pixels()
  {
    pixels_.clear();
    pixels_.resize((xr_ + 1) * yr_); // each row has (xr_ + 1) characters including newline
    for (std::size_t y = 1; y <= yr_; ++y) {
      pixels_[(xr_ + 1) * y - 1] = '\n';
    }
  }

public:
  
  explicit RayMarcher(std::size_t xr, std::size_t yr)
    : xr_(xr), yr_(yr)
  {
    time_origin_ = Clock::now();
    reset_pixels();
  }

  void draw()
  {
    time_ = std::chrono::duration_cast<std::chrono::milliseconds>(
      Clock::now() - time_origin_
    ).count() * 1.0E-3;

    for (std::size_t y = 0; y < yr_; ++y) {
      for (std::size_t x = 0; x < xr_; ++x) {
        // skipping newlines
        pixels_[(xr_ + 1) * y + x] = calculate(x, y);
      }
    }
  }

  void print(std::ostream& os = std::cout)
  {
    draw();

    static const char* move_to_topleft = "\033[0;0f";
    os << move_to_topleft << pixels_;
  }
};

int main()
{
  auto ray_marcher = std::make_unique<RayMarcher>(128, 64);

  while (true) {
    ray_marcher->print();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  return 0;
}
