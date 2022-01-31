#pragma once

#include <vector>
#include <cmath>
#include <tuple>

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
struct computeAccelerationWithGravity
{
    std::tuple<T, T, T> operator()(const int idx, Dataset &d)
    {
        const T G = d.g;
        const T ax = -(d.grad_P_x[idx] - G * d.fx[idx]);
        const T ay = -(d.grad_P_y[idx] - G * d.fy[idx]);
        const T az = -(d.grad_P_z[idx] - G * d.fz[idx]);
        return std::make_tuple(ax, ay, az);
    }
};

template <typename T, class Dataset>
struct computeAcceleration
{
    std::tuple<T, T, T> operator()(const int idx, Dataset &d)
    {
        return std::make_tuple(-d.grad_P_x[idx], -d.grad_P_y[idx], -d.grad_P_z[idx]);
    }
};

template <typename T, class FunctAccel, class Dataset>
void computePositionsImpl(const Task &t, Dataset &d, const cstone::Box<T>& box)
{
    FunctAccel accelFunct;

    size_t numParticles = t.size();

    const T *dt = d.dt.data();
    const T *du = d.du.data();

    T *x = d.x.data();
    T *y = d.y.data();
    T *z = d.z.data();
    T *vx = d.vx.data();
    T *vy = d.vy.data();
    T *vz = d.vz.data();
    T *x_m1 = d.x_m1.data();
    T *y_m1 = d.y_m1.data();
    T *z_m1 = d.z_m1.data();
    T *u = d.u.data();
    T *du_m1 = d.du_m1.data();
    T *dt_m1 = d.dt_m1.data();

#pragma omp parallel for
    for (size_t pi = 0; pi < numParticles; pi++)
    {
        int i = pi + t.firstParticle;
        T ax, ay, az;
        std::tie(ax, ay, az) = accelFunct(i, d);

#ifndef NDEBUG
        if (std::isnan(ax) || std::isnan(ay) || std::isnan(az))
            printf("ERROR::UpdateQuantities(%d) acceleration: (%f %f %f)\n", i, ax, ay, az);
#endif

        // Update positions according to Press (2nd order)
        T deltaA = dt[i] + 0.5 * dt_m1[i];
        T deltaB = 0.5 * (dt[i] + dt_m1[i]);

        T valx = (x[i] - x_m1[i]) / dt_m1[i];
        T valy = (y[i] - y_m1[i]) / dt_m1[i];
        T valz = (z[i] - z_m1[i]) / dt_m1[i];

#ifndef NDEBUG
        if (std::isnan(valx) || std::isnan(valy) || std::isnan(valz))
        {
            printf("ERROR::UpdateQuantities(%d) velocities: (%f %f %f)\n", i, valx, valy, valz);
            printf("ERROR::UpdateQuantities(%d) x, y, z, dt_m1: (%f %f %f %f)\n", i, x[i], y[i], z[i], dt_m1[i]);
        }
#endif

        vx[i] = valx + ax * deltaA;
        vy[i] = valy + ay * deltaA;
        vz[i] = valz + az * deltaA;

        x_m1[i] = x[i];
        y_m1[i] = y[i];
        z_m1[i] = z[i];

        // x[i] = x + dt[i] * valx + ax * dt[i] * deltaB;
        x[i] += dt[i] * valx + (vx[i] - valx) * dt[i] * deltaB / deltaA;
        y[i] += dt[i] * valy + (vy[i] - valy) * dt[i] * deltaB / deltaA;
        z[i] += dt[i] * valz + (vz[i] - valz) * dt[i] * deltaB / deltaA;

        if (box.pbcX() && x[i] < box.xmin())
        {
            x[i] += box.lx();
            x_m1[i] += box.lx();
        }
        else if (box.pbcX() && x[i] > box.xmax())
        {
            x[i] -= box.lx();
            x_m1[i] -= box.lx();
        }
        if (box.pbcY() && y[i] < box.ymin())
        {
            y[i] += box.ly();
            y_m1[i] += box.ly();
        }
        else if (box.pbcY() && y[i] > box.ymax())
        {
            y[i] -= box.ly();
            y_m1[i] -= box.ly();
        }
        if (box.pbcZ() && z[i] < box.zmin())
        {
            z[i] += box.lz();
            z_m1[i] += box.lz();
        }
        else if (box.pbcZ() && z[i] > box.zmax())
        {
            z[i] -= box.lz();
            z_m1[i] -= box.lz();
        }

        // Update the energy according to Adams-Bashforth (2nd order)
        deltaA = 0.5 * dt[i] * dt[i] / dt_m1[i];
        deltaB = dt[i] + deltaA;

        u[i] += du[i] * deltaB - du_m1[i] * deltaA;

#ifndef NDEBUG
        if (std::isnan(u[i]) || u[i] < 0.0)
            printf("ERROR::UpdateQuantities(%d) internal energy: u %f du %f dB %f du_m1 %f dA %f\n", i, u[i], du[i], deltaB, du_m1[i],
                   deltaA);
#endif

        du_m1[i] = du[i];
        dt_m1[i] = dt[i];
    }
}

template <typename T, class FunctAccel, class Dataset>
void computePositions(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<T>& box)
{
    for (const auto &task : taskList)
    {
        computePositionsImpl<T, FunctAccel>(task, d, box);
    }
}

} // namespace sph
} // namespace sphexa
