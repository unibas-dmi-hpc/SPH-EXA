//
// Created by Noah Kubli on 04.04.23.
//

#ifndef SPHEXA_STAR_FORMATION_HPP
#define SPHEXA_STAR_FORMATION_HPP

namespace star_formation {

template <typename T>
struct StarFormer {
    const T temp_max{30000.};
    const T rho_overdensity{0.};
    const T t_starform_min{0.};
    const T c_star{0.1};
    const T m_star{0.0};

    void form(const T& rho, const T& temp, T mass, const T& dt, const T& time)
    {
        constexpr T G{1.0};
        const T t_dynamical = 1. / std::sqrt(4. * M_PI * G * rho);
        if (temp > temp_max) return;
        if (rho < rho_overdensity) return;
        const T delta_t = std::max(dt, t_starform_min);
        const float p = 1.0 - std::exp(c_star * delta_t / t_dynamical);
        const double r_random = std::rand() / (double)RAND_MAX;
        if (mass / m_star < r_random / p) return;
        _createStarParticle(time, m_star, m_star);
        mass -= m_star;

    }
    void _createStarParticle(const T& time, const T& mass, const T& m_form) {};
};

}

#endif // SPHEXA_STAR_FORMATION_HPP
