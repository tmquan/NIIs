# NIIs
Neural Interactive Ingredients

Nontrainable
  | Name        | Type         | Params
---------------------------------------------
0 | viewer      | PictureModel | 0
1 | opacity_net | Sequential   | 32.4 M
2 | clarity_net | Sequential   | 11.0 M
3 | density_net | Sequential   | 32.4 M
4 | frustum_net | Sequential   | 18.1 M
5 | l1loss      | L1Loss       | 0
---------------------------------------------

Pixelshuffle
  | Name        | Type         | Params
---------------------------------------------
0 | viewer      | PictureModel | 0
1 | opacity_net | Sequential   | 19.9 M
2 | clarity_net | Sequential   | 6.1 M
3 | density_net | Sequential   | 19.9 M
4 | frustum_net | Sequential   | 18.1 M
5 | l1loss      | L1Loss       | 0
---------------------------------------------


