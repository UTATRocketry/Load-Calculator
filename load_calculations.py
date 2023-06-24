#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 00:00:00 2021
@author: Fred Chun

Version as of 23 of May 2023
@editor: Nat Espinosa
"""

import argparse
from ast import arg
import numpy as np
import pandas as pd
from plotly import graph_objs as go
# import plotly.io as pio

# pio.renderers.default = "pdf"


def main():

    # See details at
    # http://www.aspirespace.org.uk/downloads/Rocket%20vehicle%20loads%20and%20airframe%20design.pdf

    # Rocket's documentation
    parser = argparse.ArgumentParser()
    parser.add_argument('--Rocket_info', type=str, help='input desired rockets main information file', required=True)
    parser.add_argument('--Rocket_COP', type=str, help='input desired rockets COP file', required=True)
    parser.add_argument('--Rocket_CL', type=str, help='input desired rockets lift coefficient file', required=True)
    parser.add_argument('--Open_Rocket', type=str, help='input desired rockets OpenRocket information file', required=True)
    args = parser.parse_args()

    rocket_info_df = pd.read_csv(args.Rocket_info, sep=",", header=None, skiprows=1, names=["Length", "Reference Area", "Rocket_part", "mass_m", "x_cm"])
    rocket_info_df["Length"] = pd.to_numeric(rocket_info_df["Length"], errors='coerce')
    rocket_info_df["Reference Area"] = pd.to_numeric(rocket_info_df["Reference Area"], errors='coerce')  

    # Length [m] Note: please double check that the information is in [cm]
    length_l = rocket_info_df.iloc[0, 0]

    # Reference area
    s = np.pi * ((rocket_info_df.iloc[0, 1]/2)**2)

    # Rocket Parts
    rocket_parts_df = rocket_info_df.iloc[:, [2, 3, 4]]

    # AeroLab data
    aerolab_x_cp_df = pd.read_csv(args.Rocket_COP, sep="\s+", header=None, skiprows=1)
    aerolab_cl_alpha_df = pd.read_csv(args.Rocket_CL, sep="\s+", header=None, skiprows=1)

    # Max q data       --> manually change <--    
    open_rocket_df = pd.read_csv(args.Open_Rocket, sep=',', header=None, skiprows=1)
    
    rho_inf = open_rocket_df.iloc[0,0] # density [kg m^-3]
    velocity_v_inf = open_rocket_df.iloc[0,1] # velocity [m s^-1]
    velocity_v_wind_gust = open_rocket_df.iloc[0,2] # wind gust velocity [m s^-1]  

    angle_of_attack_alpha = np.arctan(velocity_v_wind_gust / velocity_v_inf)  # [rad]

    # Lift coefficient curve slopes [rad^-1]
    nose_body_cl_alpha = aerolab_cl_alpha_df.iloc[-1, 2]
    wing_cl_alpha = aerolab_cl_alpha_df.iloc[-1, 3]
    body_wing_cl_alpha = aerolab_cl_alpha_df.iloc[-1, 4]

    # Lift coefficients
    nose_body_cl = nose_body_cl_alpha * angle_of_attack_alpha
    wing_cl = wing_cl_alpha * angle_of_attack_alpha
    body_wing_cl = body_wing_cl_alpha * angle_of_attack_alpha

    # Lifts [N]
    nose_body_lift_l = 0.5 * rho_inf * (velocity_v_inf ** 2) * s * nose_body_cl
    wing_lift_l = 0.5 * rho_inf * (velocity_v_inf ** 2) * s * wing_cl
    body_wing_lift_l = 0.5 * rho_inf * (velocity_v_inf ** 2) * s * body_wing_cl

    rocket_lift_l = nose_body_lift_l + wing_lift_l + body_wing_lift_l

    # Centres of pressure [m]
    nose_body_x_cp = aerolab_x_cp_df.iloc[-1, 2] / 1000
    wing_x_cp = aerolab_x_cp_df.iloc[-1, 3] / 1000
    body_wing_x_cp = aerolab_x_cp_df.iloc[-1, 4] / 1000

    # Masses [kg]
    rocket_parts_mass_m = rocket_parts_df["mass_m"]

    rocket_mass_m = sum(rocket_parts_mass_m)

    # Centres of mass [m]
    rocket_parts_x_cm = rocket_parts_df["x_cm"]

    rocket_x_cm = sum(rocket_parts_mass_m * rocket_parts_x_cm) / sum(
        rocket_parts_mass_m)

    # Moments of inertia [kg m^2]
    rocket_parts_i = rocket_parts_mass_m * (
            rocket_parts_x_cm - rocket_x_cm) ** 2

    rocket_i = sum(rocket_parts_i)

    # Moments [N m]
    nose_body_moment_m = nose_body_lift_l * (nose_body_x_cp - rocket_x_cm)
    wing_moment_m = wing_lift_l * (wing_x_cp - rocket_x_cm)
    body_wing_moment_m = body_wing_lift_l * (body_wing_x_cp - rocket_x_cm)

    rocket_moment_m = nose_body_moment_m + wing_moment_m + body_wing_moment_m

    # Angular acceleration [rad s^-2]
    angular_acceleration_alpha = rocket_moment_m / rocket_i

    # Accelerations [m s^-2]
    rocket_parts_a = (rocket_lift_l / rocket_mass_m) + (
            angular_acceleration_alpha * (rocket_parts_x_cm - rocket_x_cm))

    # Point loads [N]
    rocket_parts_p = -(rocket_parts_mass_m * rocket_parts_a)

    # Load diagram
    loads_d = {
        "x": np.array(
            rocket_parts_x_cm.tolist() + [nose_body_x_cp, wing_x_cp,
                                          body_wing_x_cp] + [0, length_l]),
        "p": np.array(
            rocket_parts_p.tolist() + [nose_body_lift_l, wing_lift_l,
                                       body_wing_lift_l] + [0, 0]),
    }
    loads_df = pd.DataFrame(data=loads_d).sort_values(by=["x"])

    loads_x = loads_df["x"]  # [m]
    loads_p = loads_df["p"]  # [N]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=loads_x, y=loads_p))
    fig.update_layout(xaxis_range=[0, length_l], xaxis_title="Position, x [m]",
                      yaxis_title="Load, P [N]")
    fig.update_traces(marker=dict(line=dict(width=10, color='blue')))
    fig.show()

    # Shear diagram
    shears_d = {
        "x": np.repeat(np.copy(loads_x), 2)[1:-1],
        "shear_v": np.repeat(np.add.accumulate(np.copy(loads_p)), 2)[:-2],
    }
    shears_df = pd.DataFrame(data=shears_d)

    shears_x = shears_df["x"]  # [m]
    shears_shear_v = shears_df["shear_v"]  # [N]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shears_x, y=shears_shear_v))
    fig.update_layout(xaxis_range=[0, length_l], xaxis_title="Position, x [m]",
                      yaxis_title="Shear, V [N]")
    fig.show()

    # Moment diagram
    moments_d = {
        "x": np.copy(loads_x),
        "moment_m": np.concatenate((np.zeros(1), np.add.accumulate(
            np.concatenate(
                (np.diff(np.copy(loads_x)), np.zeros(1))) * np.add.accumulate(
                np.copy(loads_p)))))[:-1],
    }
    moments_df = pd.DataFrame(data=moments_d)

    moments_x = moments_df["x"]  # [m]
    moments_moment_m = moments_df["moment_m"]  # [N m]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=moments_x, y=moments_moment_m))
    fig.update_layout(xaxis_range=[0, length_l], xaxis_title="Position, x [m]",
                      yaxis_title="Moment, M [N m]")
    fig.show()

    return True
    

if __name__ == '__main__':
    main()

