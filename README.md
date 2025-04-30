# Modelling

This set of notebooks breaks the problem of modelling the *Deterministic* Boussinesq equations

$$
\begin{align}
  \frac{D \boldsymbol{U} }{Dt} &= -\nabla P + Ri_B \, B \boldsymbol{\hat{z}} + Re^{-1} \Delta \boldsymbol{U} + \boldsymbol{U}/\langle |\boldsymbol{U}|^2 \rangle, \\
    \frac{D B }{Dt}              &= -W + Pe^{-1} \Delta B, \\  
    \nabla \cdot \boldsymbol{U}  &= 0, 
\end{align}
$$

in terms of the time evolution of the PDF $f_{\boldsymbol{Y}}$ for the random variables $\boldsymbol{Y} = (W,B,Z)$ using the probabilistic equation

$$
\begin{equation}
    \frac{\partial f_{\boldsymbol{Y}} }{\partial t} + \frac{\partial }{\partial z} \left( w f_{\boldsymbol{Y}} \right) =   
    - \frac{\partial }{\partial b} \left( \left[ -w + Pe^{-1} \mathbb{E}_{\boldsymbol{Y}}[ \Delta B ] \right] f_{\boldsymbol{Y}} \right) 
    - \frac{\partial }{\partial w} \left( \left[ \mathbb{E}_{\boldsymbol{Y}}[ Ri_B B -\nabla_Z P] +  \mathbb{E}_{\boldsymbol{Y}}[ \frac{W}{\langle |\boldsymbol{U}|^2 \rangle}] + Re^{-1} \mathbb{E}_{\boldsymbol{Y}}[ \Delta W ] \right] f_{\boldsymbol{Y}} \right). %
\end{equation}
$$

into a hierarchial set of notebooks.


## Directory Structure

### 1D Models
The `1D_Models` directory contains Jupyter notebooks that simulate one-dimensional systems presumed to model $f_B$. These models focus on simpler scenarios and are divided into two categories:
- **IEM**:
    - `Part1_IEM_fB.ipynb`: Implements the interaction by exchange with the mean (IEM) model for a one-dimensional system. For more details on the IEM model, refer to [Pope (1985)](https://doi.org/10.1017/S0022112085000942).
    - `Part1_IEM_Particle_fB.ipynb`: Extends the IEM model to include particle-based simulations.
- **Mapping**:
    - `Part1_Mapping_fB.ipynb`: Implements the mapping closure for a one-dimensional system. For more information on the mapping closure, see [Klimenko (2005)](https://doi.org/10.1016/j.ces.2005.01.035).
    - `Part1_Mapping_Particle_fB.ipynb`: Combines mapping closure with particle-based simulations.

### 2D Models
The `2D_Models` directory includes Jupyter notebooks for two-dimensional systems, categorized by the random variables modelled:
- **BW**: contains Jupyter notebooks that simulate two-dimensional systems presumed to model $f_{BW}$
    - **IEM**:
        - `Part2_IEM_2d_fBW.ipynb`: Simulates the IEM model for a 2D system with buoyancy and width variables. For more details on the IEM model, refer to [Pope (1985)](https://doi.org/10.1017/S0022112085000942).
        - `Part2_IEM_particle_fBW.ipynb`: Extends the IEM model to include particle-based simulations.
    - **Mapping**:
        - `Part2_Mapping_2d_fBW.ipynb`: Implements the mapping closure for a 2D system. For more information on the mapping closure, see [Klimenko (2005)](https://doi.org/10.1016/j.ces.2005.01.035).
        - `Part2_Mapping_fBW.ipynb`: Focuses on mapping closure for buoyancy and width variables.
- **BZ**: contains Jupyter notebooks that simulate two-dimensional systems presumed to model $f_{BZ}$ where $Z$ must be treated differently to $W$ and $B$
    - **IEM**:
        - `Part3_IEM_2d_fBZ_Layered.ipynb`: Implements the IEM model for a 2D system with buoyancy and height variables, focusing on layered structures. For more details on the IEM model, refer to [Pope (1985)](https://doi.org/10.1017/S0022112085000942).
    - **Mapping**:
        - `Part3_Mapping_2d_fBZ_Layered.ipynb`: Implements the mapping closure for a 2D system with buoyancy and height variables, using a layered approach. For more information on the mapping closure, see [Klimenko (2005)](https://doi.org/10.1016/j.ces.2005.01.035).

### 3D Models
The `3D_Models` directory contains Jupyter notebooks for full three-dimensional system $f_{WBZ}$:
- **IEM**:
    - `Part4_IEM_3d_fWBZ.ipynb`: Implements the IEM model for a 3D system, focusing on buoyancy $B$, vertical velocity $W$, and the vertical coordinate $Z$ variables. For more details on the particle implementation of the IEM model for this case, refer to [Pope (1985)](https://doi.org/10.1017/S0022112085000942).

Each directory is structured to include relevant datasets, scripts, and documentation to support the models.