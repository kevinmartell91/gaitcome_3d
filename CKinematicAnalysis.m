classdef CKinematicAnalysis
    %CKINEMATICANALYSIS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
         v
    end
    
    methods (Access = public)
        %%%%%%%% Save 3D points to JSON FORMAT 
        function SavePoints3DToJsonFormatSample (POINT3D , INDEX,lkne_x)

        % x = [40;43;
        % y = [43;69];
        % z = [10;41];

        % jsonencode(table(x,y,z))
        %  =   '[{"x":"40","y":43, "z":10},{"x":"43","y":69, "z":41}]'
        %
        end

        %% Methods that turn 3DPoints into dictionaries
        function newMapDicPoint3D = GetDicPoint3D(POINT3D)
            
            [m n] = size(POINT3D);
            keySet   = {'x','y','z'};
            
            for i = 1:n
                valueSet(i) = POINT3D(1,i)
            end
            
            newMapDicPoint3D = containers.Map(keySet,valueSet);

        end

        function [lbwt lfwt ltrc lkne lank lhee lteo] = GetArrayDicPoints3D(POINTS3D)
    
            lteo = this.GetDicPoint3D(POINTS3D(1,:));   
            lhee = this.GetDicPoint3D(POINTS3D(2,:));
            lank = this.GetDicPoint3D(POINTS3D(3,:));
            lkne = this.GetDicPoint3D(POINTS3D(4,:));
            ltrc = this.GetDicPoint3D(POINTS3D(5,:));
            lfwt = this.GetDicPoint3D(POINTS3D(6,:));
            lbwt = this.GetDicPoint3D(POINTS3D(7,:)); % lbwt( 1 , 1=x | 2=y | 3=z);

        end

        %% Methods that calculate angles
        function [ANG_HIP  ANG_PEL ANG_KNE ANG_ANK] = CalculateAngles(POINTS3D)

           [lbwt lfwt ltrc lkne lank lhee lteo] = this.GetArrayDicPoints3D(POINTS3D);

           ANG_HIP = this.CalculateKneeAnglesSagittal(ltrc, lkne, lank);
           ANG_PEL = this.CalculatePelvisAnglesSagittal(lbwt, lfwt);
           ANG_KNE = this.CalculateKneeAnglesSagittal(ltrc, lkne, lank);
           ANG_ANK = this.CalculateKneeAnglesSagittal(ltrc, lkne, lank);
           trop =32 ;
           % ... 

        end

        function ang = CalculatePelvisAnglesSagittal(LBWT, LFWT)
            P_TOP  = [LBWT('x'), LBWT('y')];  
            P_CENT = [LFWT('x'), LFWT('y')];
            P_BOTT = [  -9999  , LFWT('y')];

            ang = this.CalculateAngle3DPoints(P_TOP, P_CENT, P_BOTT, 'relative');
        end

        function ang = CalculateKneeAnglesSagittal(LTRC, LKNE, LANK)
            P_TOP  = [LTRC('x'), LTRC('y')];  
            P_CENT = [LKNE('x'), LKNE('y')];
            P_BOTT = [LANK('x'), LANK('y')];

            ang = CalculateAngle3Points(P_TOP, P_CENT, P_BOTT, 'relative');
        end

        function angle = CalculateAngle3DPoints(P_TOP, P_CENT, P_BOTT, TYPE)
        % calculates the angle between the lines from P0 to P1 and P0 to P2.
        %     P0 = [x0, y0];  
        %     P1 = [x1, y1];
        %     P2 = [x2, y2];
            n1 = (P_BOTT - P_CENT) / norm(P_BOTT - P_CENT);  % Normalized vectors
            n2 = (P_TOP - P_CENT) / norm(P_TOP - P_CENT);
        %     angle1 = acos(dot(n1, n2));                        % Instable at (anti-)parallel n1 and n2
        %     angle2 = asin(norm(cropss(n1, n2));                % Instable at perpendiculare n1 and n2
            angle = atan2(norm(det([n2; n1])), dot(n1, n2)) * 180/pi ;  % Stable
            if strcmp (TYPE,'relative') 

        %     elseif TYPE == 'absolute' 
                angle = 180 - angle ;  % Stable
            end
        end
        
        function r = plus(o1,o2)
              r = o1 + o2;
           end
    end
    
end

