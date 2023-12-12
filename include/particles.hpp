#ifndef _PARTICLES_HPP_
#define _PARTICLES_HPP_

#include <vector>
#include <string>
#include <fstream>
#include <functional>

class Particles{
    // Private storage for things that really shouldn't be changed outside of the class.
    std::vector<std::vector<double>> velocities, accelerations;
    std::vector<double> masses;
    int N_parts;

    public:
        // Public storage for the positions to make it easier to have different acceleration functions.
        // Still really shouldn't be changed outside of the class
        std::vector<std::vector<double>> positions;

        Particles(std::vector<std::vector<double>> &pos, std::vector<std::vector<double>> &vel, std::vector<double> &mass,
                  std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)> accel) {
            this->positions = pos;
            this->velocities = vel;
            this->masses = mass;
            this->N_parts = pos.size();
            this->accelerations = accel(pos);
        }

        // Update the particle positions, and velocities using the Velocity Verlet algorithm.
        void update(std::function<std::vector<std::vector<double>>(std::vector<std::vector<double>>)> accel, double dt) {
            #pragma omp parallel for
            for (int i = 0; i < this->N_parts; ++i) {
                this->positions[i][0] += this->velocities[i][0]*dt + 0.5*this->accelerations[i][0]*dt*dt;
                this->positions[i][1] += this->velocities[i][1]*dt + 0.5*this->accelerations[i][1]*dt*dt;
                this->positions[i][2] += this->velocities[i][2]*dt + 0.5*this->accelerations[i][2]*dt*dt;
                this->velocities[i][0] += 0.5*this->accelerations[i][0]*dt;
                this->velocities[i][1] += 0.5*this->accelerations[i][1]*dt;
                this->velocities[i][2] += 0.5*this->accelerations[i][2]*dt;
            }

            this->accelerations = accel(this->positions);

            #pragma omp parallel for
            for (int i = 0; i < this->N_parts; ++i) {
                this->velocities[i][0] += 0.5*this->accelerations[i][0]*dt;
                this->velocities[i][1] += 0.5*this->accelerations[i][1]*dt;
                this->velocities[i][2] += 0.5*this->accelerations[i][2]*dt;
            }
        }

        void snapshot(std::string snapshot_name) {
            std::ofstream fout(snapshot_name, std::ios::out | std::ios::binary);
            fout.write((char *) &this->N_parts, sizeof(int));
            fout.write((char *) this->positions.data(), this->N_parts*3*sizeof(double));
            fout.write((char *) this->velocities.data(), this->N_parts*3*sizeof(double));
            fout.close();
        }
};

#endif