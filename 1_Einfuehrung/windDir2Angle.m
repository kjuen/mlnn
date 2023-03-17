function angle = windDir2Angle(windDirCat)
   %windDir2Angle: Windrichtung in Bogenmaß umrechnen
   angle = nan(size(windDirCat)); 
   angle(windDirCat=='N') = 0; 
   angle(windDirCat=='NNE') = pi / 8; 
   angle(windDirCat=='NE') = 2*pi / 8; 
   angle(windDirCat=='ENE') = 3*pi / 8; 
   angle(windDirCat=='E') = 4*pi / 8;
   angle(windDirCat=='ESE') = 5*pi / 8;
   angle(windDirCat=='SE') = 6*pi / 8;
   angle(windDirCat=='SSE') = 7*pi / 8;
   angle(windDirCat=='S') = 8*pi / 8;
   angle(windDirCat=='SSW') = 9*pi / 8;
   angle(windDirCat=='SW') = 10*pi / 8;
   angle(windDirCat=='WSW') = 11*pi / 8;
   angle(windDirCat=='W') = 12*pi / 8;
   angle(windDirCat=='WNW') = 13*pi / 8;
   angle(windDirCat=='NW') = 14*pi / 8;
   angle(windDirCat=='NNW') = 15*pi / 8;
   end

