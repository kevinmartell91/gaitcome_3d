classdef CKinematicAnalysis
    %CKINEMATICANALYSIS Provides methods for kinematic analysis of 3D points.
    
    properties
         v % General property, purpose not specified
    end
    
    methods (Access = public)
        % Save 3D points in JSON format (sample method, not fully implemented)
        function SavePoints3DToJsonFormatSample(this, POINT3D, INDEX, lkne_x)
            % Example of how to encode points into JSON format
            % Not implemented
        end

        % Convert 3D point to a dictionary with 'x', 'y', 'z' keys
        function newMapDicPoint3D = GetDicPoint3D(this, POINT3D)
            [~, n] = size(POINT3D); % Get number of columns in POINT3D
            keySet = {'x','y','z'}; % Define keys for map
            
            for i = 1:n
                valueSet(i) = POINT3D(1,i); % Assign values from POINT3D
            end
            
            newMapDicPoint3D = containers.Map(keySet,valueSet); % Create map
        end

        % Convert array of 3D points to dictionaries for each point
        function [lbwt, lfwt, ltrc, lkne, lank, lhee, lteo] = GetArrayDicPoints3D(this, POINTS3D)
            % Convert each row of POINTS3D to a dictionary
            lteo = this.GetDicPoint3D(POINTS3D(1,:));   
            lhee = this.GetDicPoint3D(POINTS3D(2,:));
            lank = this.GetDicPoint3D(POINTS3D(3,:));
            lkne = this.GetDicPoint3D(POINTS3D(4,:));
            ltrc = this.GetDicPoint3D(POINTS3D(5,:));
            lfwt = this.GetDicPoint3D(POINTS3D(6,:));
            lbwt = this.GetDicPoint3D(POINTS3D(7,:));
        end

        % Calculate angles based on 3D points
        function [ANG_HIP, ANG_PEL, ANG_KNE, ANG_ANK] = CalculateAngles(this, POINTS3D)
            % Retrieve dictionaries for each point
            [lbwt, lfwt, ltrc, lkne, lank, lhee, lteo] = this.GetArrayDicPoints3D(POINTS3D);

            % Calculate angles using specific methods
            ANG_HIP = this.CalculateKneeAnglesSagittal(ltrc, lkne, lank);
            ANG_PEL = this.CalculatePelvisAnglesSagittal(lbwt, lfwt);
            ANG_KNE = this.CalculateKneeAnglesSagittal(ltrc, lkne, lank);
            ANG_ANK = this.CalculateKneeAnglesSagittal(ltrc, lkne, lank);
        end

        % Calculate pelvis angle in sagittal plane
        function ang = CalculatePelvisAnglesSagittal(this, LBWT, LFWT)
            % Define points for angle calculation
            P_TOP  = [LBWT('x'), LBWT('y')];  
            P_CENT = [LFWT('x'), LFWT('y')];
            P_BOTT = [-9999, LFWT('y')]; % Placeholder value for P_BOTT x-coordinate

            % Calculate angle
            ang = this.CalculateAngle3DPoints(P_TOP, P_CENT, P_BOTT, 'relative');
        end

        % Calculate knee angle in sagittal plane
        function ang = CalculateKneeAnglesSagittal(this, LTRC, LKNE, LANK)
            % Define points for angle calculation
            P_TOP  = [LTRC('x'), LTRC('y')];  
            P_CENT = [LKNE('x'), LKNE('y')];
            P_BOTT = [LANK('x'), LANK('y')];

            % Calculate angle
            ang = this.CalculateAngle3DPoints(P_TOP, P_CENT, P_BOTT, 'relative');
        end

        % Calculate angle between three points
        function angle = CalculateAngle3DPoints(this, P_TOP, P_CENT, P_BOTT, TYPE)
            % Calculate normalized vectors
            n1 = (P_BOTT - P_CENT) / norm(P_BOTT - P_CENT);
            n2 = (P_TOP - P_CENT) / norm(P_TOP - P_CENT);
            % Calculate angle using stable method
            angle = atan2(norm(det([n2; n1])), dot(n1, n2)) * 180/pi;
            if strcmp(TYPE,'relative')
                angle = 180 - angle; % Adjust angle for relative measurement
            end
        end
        
        % Overload plus operator for class objects
        function r = plus(o1, o2)
            r = o1 + o2; % Example of operator overloading, not specific
        end
    end
    
end