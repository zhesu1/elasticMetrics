# elasticMetrics

Tools for Geometric shape analysis of spherical surfaces with first order elastic metrics based on the work by Martin Bauer, Eric Klassen, Hamid Laga, Stephen C. Preston and Zhe Su.

## What is it?

This code provides tools for geometric shape analysis on spherical surfaces with first order elastic metrics. 
It is able to factor out reparametrizations, translations and rotations. The following are two examples of geodesics between unparametrized surfaces presented in the paper "[Shape Analysis of Surfaces Using General Elastic Metrics](https://arxiv.org/abs/1910.02045)".

<img align="left" src="https://github.com/zhesu1/Figures/raw/master/registeredcat12_geo_unpara_1_1_-1_0_deg7_degv7_T13.png" width="800"><br clear="both"/> 

<img align="left" src="https://github.com/zhesu1/Figures/raw/master/registeredhorse02_geo_unpara_1_1_-1_0_deg7_degv7_T13.png" width="790"><br clear="both"/>

For details we refer to our papers

```css
@article{BKPS2018,
    title={A Diffeomorphism-Invariant Metric on the Space of Vector-Valued One-Forms}, 
    author={Martin Bauer, Eric Klassen, Stephen C. Preston, Zhe Su},
    journal={arXiv:1812.10867},
    year={2018},  
}

@article{SBPLK2019, 
    title={Shape Analysis of Surfaces Using General Elastic Metrics},
    author={Zhe Su, Martin Bauer, Stephen C. Preston, Hamid Laga, Eric Klassen},
    journal={arXiv:1910.02045},
    year={2019},  
}
```

If you use our code in your work please cite our papers.

## Packages

Please install the following packages

* Pytorch: [https://pytorch.org/](https://pytorch.org/)
* Numpy: [https://numpy.org/](https://numpy.org/)
* Scipy: [https://www.scipy.org/](https://www.scipy.org/)
* Mayavi (for plotting): [https://docs.enthought.com/mayavi/mayavi/](https://docs.enthought.com/mayavi/mayavi/)

The code was tested on jupyter notebook.

## Usage

See the files "Calculate_geodesic_para", "Calculate_geodesic_unpara" and "Calculate_geodesic_unparaModSO3" for examples of how to use the code. Each surface should be represented using spherical coordinates as a function f:[0, &pi;] x [0, 2&pi;] &rarr; R<sup>3</sup> 
such that f(0, &theta;) = f(0, 0), f(&pi;, &theta;) = f(&pi;, 0) and f(&phi;, 0) = f(&phi;, 2&pi;). The code can be used directly for surfaces with resolution 50 x 99 (the numbers of discrete polar and azimuthal angles) since the corresponding bases are preloaded in the folder "Bases".  For the other resolutions, one can resample the surfaces or generate the corresponding bases for surfaces using the files in the folder "generateBases" and then put the bases files in the folder "Bases". 

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)

## Contacts

* Martin Bauer (bauer at math dot fsu dot edu)
* Zhe Su (zsu at math dot fsu dot edu)
