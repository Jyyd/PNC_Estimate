%{
 * @Author: JYYD jyyd23@mails.tsinghua.edu.cn
 * @Date: 2023-11-29 16:56:14
 * @LastEditors: JYYD jyyd23@mails.tsinghua.edu.cn
 * @LastEditTime: 2024-01-09 15:02:12
 * @FilePath: \code\finalcode\PNC_estimate\diffplot.m
 * @Description: 
 * 
%}

clc;
clear;close all;

year = 2020;
Nx = 54;
Ny = 22;
fontsize = 20;
diffFlag = 0;
charLength = 1/30;

folderPath = 'code/pncEstimator-main/src/postProcessing/matdata/PNCtest/0102/4part/';
% folderPath = 'code/pncEstimator-main/src/postProcessing/matdata/pncall3/';
matFiles0 = dir(fullfile(folderPath, '*.mat'));
monthFlag = 0; % if season 0, month 1

if monthFlag==1
    fileNames = {matFiles0.name};
    pattern = '(\d+)(?=\.\w+$)'; 
    lastNumbers = regexp(fileNames, pattern, 'match', 'once');
    lastNumbers = str2double(lastNumbers);
    [~, sortIndex] = sort(lastNumbers);
    matFiles = matFiles0(sortIndex);

    for i = 1:(length(matFiles) - 1)
        fileName1 = matFiles(i).name;
        fileName2 = matFiles(i + 1).name;
        pattern = '_(\w+)\.mat'; 
                match1 = regexp(fileName1, pattern, 'tokens');
                match2 = regexp(fileName2, pattern, 'tokens');

                if ~isempty(match1) && ~isempty(match2)
                    targetString1 = match1{1}{1};
                    targetString2 = match2{1}{1};

                    parts1 = split(targetString1, '_');
                    lastPart1 = parts1{3,1};
                    parts2 = split(targetString2, '_');
                    lastPart2 = parts2{3,1};

                    file1 = fullfile(folderPath, fileName1);
                    file2 = fullfile(folderPath, fileName2);
                    loadmat1 = load(file1);
                    loadmat2 = load(file2);
                    matrix1 = loadmat1.avgConc;
                    matrix2 = loadmat2.avgConc;
                    avgConc = loadmat1.avgConc;
                    diffMatrix = matrix2 - matrix1;

                    lonNew = loadmat1.lonNew;
                    latNew = loadmat1.latNew;
                    [xnew, ynew] = meshgrid(lonNew, latNew);
                    Delta_x = mean(diff(lonNew));
                    Delta_y = abs(mean(diff(latNew)));
                    figure('position', [100 100 1000 600])
                    mapFile ='code/pncEstimator-main/data/geoshp/gadm36_CHE_0.shp';
                    roi = shaperead(mapFile);
                    names = {roi.NAME_0};
                    for regionId=1:length(roi)
                        rx = roi(regionId).X;
                        ry = roi(regionId).Y;
                        plot(rx, ry, 'k-', 'linewidth',1.5)
                        blockNum = sum(isnan(rx));
                        blockId = find(isnan(ry));
                        blockId = [0 blockId];
                        x_min = min(lonNew);
                        y_min = min(latNew);
                        Ny = size(avgConc,1);
                        Nx = size(avgConc,2);
                        maskTmp = 0;
                        for i =1:blockNum
                            % convert to image coordinates
                            ix = double((rx((blockId(i)+1):(blockId(i+1)-1)) - x_min)/Delta_x + 1);
                            iy = Ny- double((ry((blockId(i)+1):(blockId(i+1)-1)) - y_min)/Delta_y + 1);
                            maskTmp = min(maskTmp + poly2mask(ix,iy,Ny,Nx),1);
                        end
                    end

                    imagesc(lonNew, latNew, diffMatrix.*maskTmp, 'Alphadata', ...
                        maskTmp,[-5000,5000]);
                    
                    % add text
                    if strcmp(lastPart1, 'month1') && strcmp(lastPart2, 'month2')
                        text(5.7, 47.78, 'Feb. minus Jan.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month2') && strcmp(lastPart2, 'month3')
                        text(5.7, 47.78, 'Mar. minus Feb.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month3') && strcmp(lastPart2, 'month4')
                        text(5.7, 47.78, 'Apr. minus Mar.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month4') && strcmp(lastPart2, 'month5')
                        text(5.7, 47.78, 'May. minus Apr.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month5') && strcmp(lastPart2, 'month6')
                        text(5.7, 47.78, 'Jun. minus May.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month6') && strcmp(lastPart2, 'month7')
                        text(5.7, 47.78, 'Jul. minus Jun.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month7') && strcmp(lastPart2, 'month8')
                        text(5.7, 47.78, 'Aug. minus Jul.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month8') && strcmp(lastPart2, 'month9')
                        text(5.7, 47.78, 'Sep. minus Aug.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month9') && strcmp(lastPart2, 'month10')
                        text(5.7, 47.78, 'Oct. minus Sep.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month10') && strcmp(lastPart2, 'month11')
                        text(5.7, 47.78, 'Nov. minus Oct.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'month11') && strcmp(lastPart2, 'month12')
                        text(5.7, 47.78, 'Dec. minus Nov.', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    else

                    end


                    daspect([1 cos(mean(latNew)/180*pi) 1])
                    axis xy

                    % 定义自定义颜色映射的颜色列表
                    numColors = 64; % 颜色数量
                    blueColor = [0, 0, 1]; % 蓝色
                    whiteColor = [1, 1, 1]; % 白色
                    redColor = [1, 0, 0]; % 红色
                    customColormap = zeros(numColors, 3);
                    transitionLength = floor(numColors / 2); % 过渡长度，左半部分
                    customColormap(1:transitionLength, :) = interp1([0, transitionLength], [blueColor; whiteColor], 0:transitionLength-1);
                    customColormap(transitionLength+1:end, :) = interp1([0, transitionLength], [whiteColor; redColor], 0:transitionLength-1);
                    colormap(customColormap);

                    colormap;
                    shading interp;
                    cb =  colorbar;
                    cbarrow;

                    h=gcf;
                    c=get(h,'children'); % Find allchildren
                    cb=findobj(h,'Tag','Colorbar'); % Find thecolorbar children
                    cb.FontSize = fontsize;
                    cb.Ticks = linspace(-5000, 5000, 5);
                    barTicks = cb.Ticks;
                    for i=1:length(barTicks)
                        cb.FontName = 'Arial';
                    end
                    cb.Label.String = {['PNC concentration (#\cdotcm^{-3})']};

                    ylabel('Latitude (degree)');
                    xlabel('Longitude (degree)')
                    set(gca, 'FontName', 'Arial', ...
                        'FontSize', fontsize);
                    grid on; % turn on grid
                    set(gca, 'GridLineStyle', '--'); % set grid style to dashed
                    % Modify x-axis labels to include 'E'
                    xTicks = get(gca, 'XTick');
                    newXTicks = arrayfun(@(x) sprintf('%.1f°E', x), xTicks, 'UniformOutput', false);
                    xticklabels(newXTicks);

                    % Modify y-axis labels to include 'N'
                    yTicks = get(gca, 'YTick');
                    newYTicks = arrayfun(@(y) sprintf('%.1f°N', y), yTicks, 'UniformOutput', false);
                    yticklabels(newYTicks);
                    set(gcf,'PaperPositionMode','auto')

                    print(['code/finalcode/SI_figure/S3/month/Comparing '  lastPart1 ' and ' lastPart2 ],'-dpng','-r600')
                end
    end
else
    matFiles = matFiles0;
    for i = 1:length(matFiles)
        for j = i+1:length(matFiles)
            %  chek index
            if i <= length(matFiles) && j <= length(matFiles)
                % get mark string from files
                fileName1 = matFiles(i).name;
                fileName2 = matFiles(j).name;
                pattern = '_(\w+)\.mat';
                match1 = regexp(fileName1, pattern, 'tokens');
                match2 = regexp(fileName2, pattern, 'tokens');

                if ~isempty(match1) && ~isempty(match2)
                    targetString1 = match1{1}{1};
                    targetString2 = match2{1}{1};

                    parts1 = split(targetString1, '_');
                    lastPart1 = parts1{3,1};
                    parts2 = split(targetString2, '_');
                    lastPart2 = parts2{3,1};

                    file1 = fullfile(folderPath, matFiles(i).name);
                    file2 = fullfile(folderPath, matFiles(j).name);
                    loadmat1 = load(file1);
                    loadmat2 = load(file2);
                    matrix1 = loadmat1.avgConc;
                    matrix2 = loadmat2.avgConc;
                    avgConc = loadmat1.avgConc;
                    diffMatrix = matrix2 - matrix1;

                    lonNew = loadmat1.lonNew;
                    latNew = loadmat1.latNew;
                    [xnew, ynew] = meshgrid(lonNew, latNew);
                    Delta_x = mean(diff(lonNew));
                    Delta_y = abs(mean(diff(latNew)));
                    figure('position', [100 100 1000 600])
                    mapFile ='code/pncEstimator-main/data/geoshp/gadm36_CHE_0.shp';
                    roi = shaperead(mapFile);
                    names = {roi.NAME_0};
                    for regionId=1:length(roi)
                        rx = roi(regionId).X;
                        ry = roi(regionId).Y;
                        plot(rx, ry, 'k-', 'linewidth',1.5)
                        blockNum = sum(isnan(rx));
                        blockId = find(isnan(ry));
                        blockId = [0 blockId];
                        x_min = min(lonNew);
                        y_min = min(latNew);
                        Ny = size(avgConc,1);
                        Nx = size(avgConc,2);
                        maskTmp = 0;
                        for i =1:blockNum
                            % convert to image coordinates
                            ix = double((rx((blockId(i)+1):(blockId(i+1)-1)) - x_min)/Delta_x + 1);
                            iy = Ny- double((ry((blockId(i)+1):(blockId(i+1)-1)) - y_min)/Delta_y + 1);
                            maskTmp = min(maskTmp + poly2mask(ix,iy,Ny,Nx),1);
                        end
                    end

                    imagesc(lonNew, latNew, diffMatrix.*maskTmp, 'Alphadata', ...
                        maskTmp,[-5000,5000]);
                    % add text
                    if strcmp(lastPart1, 'part0') && strcmp(lastPart2, 'part1')
                        text(5.7, 47.78, 'Month 4-6 minus Month 1-3', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'part1') && strcmp(lastPart2, 'part2')
                        text(5.7, 47.78, 'Month 7-9 minus Month 4-6', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    elseif strcmp(lastPart1, 'part2') && strcmp(lastPart2, 'part3')
                        text(5.7, 47.78, 'Month 10-12 minus Month 7-9', 'FontSize', 20, 'FontName', 'Arial', 'Color', ... 
                                'black', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'top', 'FontAngle', 'italic')
                    else

                    end


                    daspect([1 cos(mean(latNew)/180*pi) 1])
                    axis xy

                    % 定义自定义颜色映射的颜色列表
                    numColors = 64; % 颜色数量
                    blueColor = [0, 0, 1]; % 蓝色
                    whiteColor = [1, 1, 1]; % 白色
                    redColor = [1, 0, 0]; % 红色
                    customColormap = zeros(numColors, 3);
                    transitionLength = floor(numColors / 2); % 过渡长度，左半部分
                    customColormap(1:transitionLength, :) = interp1([0, transitionLength], [blueColor; whiteColor], 0:transitionLength-1);
                    customColormap(transitionLength+1:end, :) = interp1([0, transitionLength], [whiteColor; redColor], 0:transitionLength-1);
                    colormap(customColormap);

                    colormap;
                    shading interp;
                    cb =  colorbar;
                    cbarrow;

                    h=gcf;
                    c=get(h,'children'); % Find allchildren
                    cb=findobj(h,'Tag','Colorbar'); % Find thecolorbar children
                    cb.FontSize = fontsize;
                    cb.Ticks = linspace(-5000, 5000, 5);
                    barTicks = cb.Ticks;
                    for i=1:length(barTicks)
                        cb.FontName = 'Arial';
                    end
                    cb.Label.String = {['PNC concentration (#\cdotcm^{-3})']};

                    ylabel('Latitude (degree)');
                    xlabel('Longitude (degree)')
                    set(gca, 'FontName', 'Arial', ...
                        'FontSize', fontsize);
                    grid on; % turn on grid
                    set(gca, 'GridLineStyle', '--'); % set grid style to dashed
                    % Modify x-axis labels to include 'E'
                    xTicks = get(gca, 'XTick');
                    newXTicks = arrayfun(@(x) sprintf('%.1f°E', x), xTicks, 'UniformOutput', false);
                    xticklabels(newXTicks);

                    % Modify y-axis labels to include 'N'
                    yTicks = get(gca, 'YTick');
                    newYTicks = arrayfun(@(y) sprintf('%.1f°N', y), yTicks, 'UniformOutput', false);
                    yticklabels(newYTicks);
                    set(gcf,'PaperPositionMode','auto')

                    print(['code/finalcode/SI_figure/S3/season/Comparing '  lastPart1 ' and ' lastPart2 ],'-dpng','-r600')
                end
            end
        end
    end
end

function h = cbarrow(options)
%% cbarrow documentation
% The cbarrow function places triangle-shaped endmembers on colorbars to
% indicate that data values exist beyond the extents of the values shown in
% the colorbar.  
% 
% This function works by creating a set of axes atop the current figure and 
% placing patch objects on the new axes.  Thus, editing a figure after
% calling cbarrow may cause some glitches.  Therefore, it is recommended to call
% cbarrow last when creating plots. 
% 
%% Syntax 
% 
%  cbarrow
%  cbarrow(Direction) 
%  cbarrow('delete')
%  h = cbarrow(...)
% 
%% Description
% 
% cbarrow places triangle-shaped endmembers on both ends of the current
% colorbar. 
% 
% cbarrow(Direction) specifies a single direction to place a colorbar end
% arrow.  Direction can be 'up', 'down', 'right', or 'left'. 
% 
% cbarrow('delete') deletes previously-created cbarrow objects. 
% 
% h = cbarrow(...) returns a handle of the axes on which cbarrow
% objects are created.  
% 
%% Example 1: Both directions
% 
% surf(peaks)
% axis tight
% colorbar 
% caxis([0 3]) 
% cbarrow
% 
%% Example 2: One direction
% 
% surf(peaks) 
% axis tight
% colorbar('southoutside') 
% colormap(brewermap(256,'*RdBu'))
% caxis([-7 7]) 
% cbarrow('right') 
% 
%% Known issues 
% This function only works once per figure.  If you have multiple subplots,
% you can only use it once, and you'll have to call cbarrow last.  Also, 
% editing plots after calling cbarrow can sometimes be a bit glitchy. 
% 
%% Author Info
% The newcolorbar function was written by Chad A. Greene of the 
% University of Texas at Austin's Institute for Geophysics (UTIG), August 2015. 
% Updated June 2016 to fix a bug in cbarrow('down'), thanks to Aodat for pointing this out. 
% http://www.chadagreene.com.
% 
% See also caxis and colorbar.
%% Error checks: 
assert(verLessThan('matlab','8.4.0')==0,'Sorry, the cbarrow function requires Matlab R2014b or later.') 
narginchk(0,1) 
%% Guess which arrows to create based on current colorbar orientation:
% Find handles of all colorbars in current figure: 
hcb = findobj(gcf,'Type','Colorbar'); 
% If no colorbars exist in current figure, create a new one: 
if isempty(hcb)
    cb = colorbar; 
else
    % Otherwise, use the most recent colorbar in the list: 
    cb = hcb(1); 
end
    
cbpos = cb.Position; 
ax1 = gca; 
ax1pos = get(ax1,'OuterPosition');
% If the colorbar is wider than it is tall, make left and right arrows: 
if cbpos(4)<cbpos(3)
    makerightarrow = true; 
    makeleftarrow = true; 
    makeuparrow = false; 
    makedownarrow = false; 
else
    % Otherwise make up and down arrows: 
    makerightarrow = false; 
    makeleftarrow = false; 
    makeuparrow = true; 
    makedownarrow = true; 
end
%% Override automatic arrow selection if user requested a specific arrow: 
if nargin>0 
    switch lower(options)
        case {'del','delete'}
            try
                h_cbarrow = findobj(gcf,'tag','cbarrow'); 
                delete(h_cbarrow); 
            end
            return
            
        case {'r','right'}
            makerightarrow = true; 
            makeleftarrow = false; 
            
        case {'l','left'}
            makeleftarrow = true; 
            makerightarrow = false; 
            
        case {'u','up'} 
            makeuparrow = true; 
            makedownarrow = false; 
            
        case {'d','down'} 
            makedownarrow = true; 
            makeuparrow = false; 
            
        otherwise
            error('Invalid input in cbarrow. Must be ''up'',''down'', ''left'', ''right'', ''delete'', or no inputs at all for automatic cbarrowing.') 
    end
end
%% Shrink position of the colorbar to allow room for arrows: 
if makerightarrow
    cbpos = cbpos + [0 0 -cbpos(4)*sqrt(2)/2 0];
    cb.Position = cbpos; 
end
if makeleftarrow
    cbpos = cbpos + [cbpos(4)*sqrt(2)/2 0 -cbpos(4)*sqrt(2)/2 0];
    cb.Position = cbpos; 
end
if makeuparrow 
    cbpos = cbpos + [0 0 0 -cbpos(3)*sqrt(2)/2];
    cb.Position = cbpos; 
end
if makedownarrow 
    cbpos = cbpos + [0 cbpos(3)*sqrt(2)/2 0 -cbpos(3)*sqrt(2)/2];
    cb.Position = cbpos; 
end
%% Create triangle arrows as patch objects in new axes: 
% Get colormap so we know what color to make the triangles: 
cm = colormap; 
% Create background axes on which to plot patch objects: 
h = axes('position',[0 0 1 1],'tag','cbarrow');
hold on
% Plot arrows: 
if makerightarrow 
    rightarrowx = (cbpos(1)+cbpos(3)) + [0 cbpos(4)*sqrt(2)/2 0 0]; 
    rightarrowy = cbpos(2) + [0 cbpos(4)/2 cbpos(4) 0]; 
    hr = patch(rightarrowx,rightarrowy,cm(end,:),'EdgeColor',cm(end,:)); 
end
if makeleftarrow
    leftarrowx = cbpos(1) + [0 -cbpos(4)*sqrt(2)/2 0 0]; 
    leftarrowy = cbpos(2) + [0 cbpos(4)/2 cbpos(4) 0]; 
    hl = patch(leftarrowx,leftarrowy,cm(1,:),'EdgeColor',cm(1,:)); 
end
if makeuparrow
    uparrowx = cbpos(1) + [0 cbpos(3) cbpos(3)/2 0]; 
    uparrowy = cbpos(2)+cbpos(4) + [0 0 cbpos(3)*sqrt(2)/2 0]; 
    hu = patch(uparrowx,uparrowy,cm(end,:),'EdgeColor',cm(end,:)); 
end
if makedownarrow
    downarrowx = cbpos(1) + [0 cbpos(3) cbpos(3)/2 0]; 
    downarrowy = cbpos(2) + [0 0 -cbpos(3)*sqrt(2)/2 0]; 
    hd = patch(downarrowx,downarrowy,cm(1,:),'EdgeColor',cm(1,:)); 
end
%% Change edge colors: 
if strcmpi(cb.Box,'on')
    
    % Get starting color and linewidth of colorbar box: 
    linecolor = cb.Color; 
    linewidth = cb.LineWidth; 
    
    % Turn off colorbar box and we'll create a new one: 
    cb.Box = 'off'; 
    
    % Edge line for left and right arrows: 
    if all([makerightarrow makeleftarrow])
        line(cbpos(1) +[0 cbpos(3) cbpos(3)+cbpos(4)*sqrt(2)/2 cbpos(3) 0 -cbpos(4)*sqrt(2)/2 0],...
            cbpos(2) + [0 0 cbpos(4)/2 cbpos(4) cbpos(4) cbpos(4)/2 0],...
            'color',linecolor,'LineWidth',linewidth)
    end
    
    % Edge line for right only: 
    if all([makerightarrow ~makeleftarrow])
        line(cbpos(1) +[0 cbpos(3) cbpos(3)+cbpos(4)*sqrt(2)/2 cbpos(3) 0 0],...
            cbpos(2) + [0 0 cbpos(4)/2 cbpos(4) cbpos(4) 0],...
            'color',linecolor,'LineWidth',linewidth)
    end
        
    % Edge line for left arrow only: 
    if all([~makerightarrow makeleftarrow])
        line(cbpos(1) +[0 cbpos(3) cbpos(3) 0 -cbpos(4)*sqrt(2)/2 0],...
            cbpos(2) + [0 0 cbpos(4) cbpos(4) cbpos(4)/2 0],...
            'color',linecolor,'LineWidth',linewidth)
    end
    
    % Edge line for up and down arrows: 
    if all([makeuparrow makedownarrow])
        line(cbpos(1) +[0 0 cbpos(3)/2 cbpos(3) cbpos(3) cbpos(3)/2 0],...
            cbpos(2) + [0 cbpos(4) cbpos(4)+cbpos(3)*sqrt(2)/2 cbpos(4) 0 -cbpos(3)*sqrt(2)/2 0],...
            'color',linecolor,'LineWidth',linewidth)
    end
    
    % Edge line for up arrow only: 
    if all([makeuparrow ~makedownarrow])
        line(cbpos(1) +[0 0 cbpos(3)/2 cbpos(3) cbpos(3) 0],...
            cbpos(2) + [0 cbpos(4) cbpos(4)+cbpos(3)*sqrt(2)/2 cbpos(4) 0 0],...
            'color',linecolor,'LineWidth',linewidth)
    end
    
    % Edge line for down arrow only: 
    if all([~makeuparrow makedownarrow])
        line(cbpos(1) +[0 0 cbpos(3) cbpos(3) cbpos(3)/2 0],...
            cbpos(2) + [0 cbpos(4) cbpos(4) 0 -cbpos(3)*sqrt(2)/2 0],...
            'color',linecolor,'LineWidth',linewidth)
    end
end
%% Clean up: 
axis off
axis([0 1 0 1]) 
% If original current axes were resized, unresize them: 
set(ax1,'OuterPosition',ax1pos)
switch lower(cb.Location) 
    case {'south','east','north','west'} 
        % Bring our arrow patch object axes to the front if the colorbar is inside the current axes: 
        axes(h)
    otherwise
        % Bring user's data axes to the front: 
        axes(ax1) 
end
% Delete output if user did not request it: 
if nargout==0
    clear h
end
end