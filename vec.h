#pragma once

#include <math.h>

#define __both__ __host__ __device__

class vec3 {
public:
	float x;
	float y;
	float z;

	inline __both__ vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	inline __both__ vec3(float t) : x(t), y(t), z(t) {}
	inline __both__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

	inline __both__ vec3 operator-() const { return vec3(-x, -y, -z); }
};

inline __both__ vec3 operator+(const vec3& v1, const vec3& v2) {
	return vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

inline __both__ vec3 operator-(const vec3& v1, const vec3& v2) {
	return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

inline __both__ vec3 operator*(const vec3& v1, const vec3& v2) {
	return vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

inline __both__ vec3 operator/(const vec3& v1, const vec3& v2) {
	return vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

inline __both__ vec3 operator*(const float s, const vec3& v) {
	return vec3(s * v.x, s * v.y, s * v.z);
}

inline __both__ vec3 operator/(const vec3& v, const float s) {
	return vec3(v.x / s, v.y / s, v.z / s);
}

inline __both__ vec3 operator*(const vec3& v, const float s) {
	return vec3(v.x * s, v.y * s, v.z * s);
}

inline __both__ vec3 cross(const vec3 a, const vec3 b)
{
	return vec3(a.y * b.z - a.z * b.y,
		-(a.x * b.z - a.z * b.x),
		a.x * b.y - a.y * b.x);
}

inline __both__ float dot(const vec3 a, const vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __both__ float len(const vec3 v)
{
	return sqrtf(dot(v, v));
}

inline __both__ vec3 normalize(const vec3 v)
{
	float l = 1.0 / len(v);
	return vec3(v.x * l, v.y * l, v.z * l);
}

__global__
void norm_vecs(const int N, vec3* xs, vec3* ys)
{
	for (int i = 0; i < N; i++)
		ys[i] = normalize(xs[i]);
}