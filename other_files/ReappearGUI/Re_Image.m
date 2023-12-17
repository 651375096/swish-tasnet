function varargout = Re_Image(varargin)
% RE_IMAGE MATLAB code for Re_Image.fig
%      RE_IMAGE, by itself, creates a new RE_IMAGE or raises the existing
%      singleton*.
%
%      H = RE_IMAGE returns the handle to a new RE_IMAGE or the handle to
%      the existing singleton*.
%
%      RE_IMAGE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RE_IMAGE.M with the given input arguments.
%
%      RE_IMAGE('Property','Value',...) creates a new RE_IMAGE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Re_Image_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Re_Image_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Re_Image

% Last Modified by GUIDE v2.5 05-Aug-2020 15:40:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Re_Image_OpeningFcn, ...
                   'gui_OutputFcn',  @Re_Image_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Re_Image is made visible.
function Re_Image_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Re_Image (see VARARGIN)

% Choose default command line output for Re_Image
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Re_Image wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Re_Image_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton_open.
function pushbutton_open_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_open (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%--------------------打开文件----------------
global SampleHead;
global SampleData;
[file,path] = uigetfile('*.lvm');
if isequal(file,0)
   disp('User selected Cancel');
   return; 
else
   disp(['User selected ', fullfile(path,file)]);  
   set(handles.edit_file,'string',fullfile(path,file));    
end


disp(['read data begin: ', datestr(now,0)]);%打印现在时间

% substr=strsplit(file,'.');
% name=substr(1);
% suffix=substr(2);
% disp(['name:',name,'suffix:',suffix]);
if strfind(file,'Head') %选择分割后的头文件，须判断数据文件
    %disp([file,' 包含Head']);
    file2=strrep(file,'Head',''); %查找并替换子字符串strrep(str,old,new)  
    if ~exist(fullfile(path,file2),'file')
        display('no sample data file');
        return;
    end
    SampleHead=fopen(fullfile(path,file));
    SampleData=load(fullfile(path,file2));
    disp(['read data end: ', datestr(now,0)]);%打印现在时间

else                      %选择数据文件，须判断分割头文件，若未分割则先分割
    %disp([file,' 不包含Head']);
    file2=strrep(file,'.lvm','Head.lvm'); 
    if ~exist(fullfile(path,file2),'file') %未含分割头文件
        %display('no data Head file');
     %--------分割文件头部分-------------
        fsource=fopen(fullfile(path,file));
        fdest=fopen(fullfile(path,file2),'a');%创建头文件
        Ln=0;
        while 1
            data=fgetl(fsource); %找到源数据文件起始的标记行
            fprintf(fdest,'%s\n',data);
            Ln=Ln+1;
            if strfind(data,'X_Value	Chan 0	Comment')
                fclose(fdest);
                break;
            end          
        end
        %d=textread(fullfile(path,file),'','headerlines', Ln) %意味着读取数据的时候跳过前Ln行。
        file3=strrep(file,'.lvm','Data.lvm'); 
        fdest2=fopen(fullfile(path,file3),'a');%创建数据文件
        while  1
            data=fgetl(fsource);
            if data==-1
                break;
            end
            data2=textscan(data,'%s %s'); 
            if isempty(data2{2})%只有一列数据
                fprintf(fdest2,'%s\n',cell2mat(data2{1,1}));  
            else        %有两列数据
                %disp(data2{1,2});
                fprintf(fdest2,'%s\n',cell2mat(data2{1,2}));                
            end
        end
        fclose all;
        delete(fullfile(path,file));%删除源文件
        movefile(fullfile(path,file3),fullfile(path,file));              
    end
    SampleHead=fopen(fullfile(path,file2));
    SampleData=load(fullfile(path,file));
    disp(['read data end: ', datestr(now,0)]);%打印现在时间
end




function edit_Pixclock_Callback(hObject, eventdata, handles)
% hObject    handle to edit_Pixclock (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_Pixclock as text
%        str2double(get(hObject,'String')) returns contents of edit_Pixclock as a double


% --- Executes during object creation, after setting all properties.
function edit_Pixclock_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_Pixclock (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_HorTotalPixs_Callback(hObject, eventdata, handles)
% hObject    handle to edit_HorTotalPixs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_HorTotalPixs as text
%        str2double(get(hObject,'String')) returns contents of edit_HorTotalPixs as a double


% --- Executes during object creation, after setting all properties.
function edit_HorTotalPixs_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_HorTotalPixs (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_HorPixles_Callback(hObject, eventdata, handles)
% hObject    handle to edit_HorPixles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_HorPixles as text
%        str2double(get(hObject,'String')) returns contents of edit_HorPixles as a double


% --- Executes during object creation, after setting all properties.
function edit_HorPixles_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_HorPixles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_VerPixles_Callback(hObject, eventdata, handles)
% hObject    handle to edit_VerPixles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_VerPixles as text
%        str2double(get(hObject,'String')) returns contents of edit_VerPixles as a double


% --- Executes during object creation, after setting all properties.
function edit_VerPixles_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_VerPixles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton_ReImage.
function pushbutton_ReImage_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_ReImage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%----------复现--------------------------
global SampleHead;
global SampleData;

XDelta=0;Samples=0;
fseek(SampleHead,0,'bof');%设定指针位于起始位置
while 1
    data=fgetl(SampleHead);
    if data==-1
         break;
    end
    if strfind(data,'Delta_X')  %找到 Delta_X 标记行    
        %disp(data);
        data2=textscan(data,'%s %f'); 
        XDelta=data2{1,2};disp(XDelta);
    end
    if strfind(data,'Samples')  %找到 Samples 标记行    
        %disp(data);
        data2=textscan(data,'%s %d'); 
        Samples=data2{1,2};disp(Samples);
    end   
    if strfind(data,'X_Value	Chan 0	Comment')    %找到起始的标记行
        %disp(data);
        break;
    end
end
L=length(SampleData);
if L<Samples
    disp('N<Samples');
    N=L;            %采样点数
else
    N=Samples;      %采样点数
end 

% Pixclock=40e+6; %像素时钟
% HorTotalPixs=1056; %水平像素点数 Pixels
% HorPixles=800; %水平分辨率
% VerPixles=600; %垂直分辨率
%%%%像素时钟--------------------
Pixclock=str2num(get(handles.edit_Pixclock,'string'));
if Pixclock==0 
    error('Pixclock null');
end
%%%%%总的水平像素点数-----------
HorTotalPixs=str2num(get(handles.edit_HorTotalPixs,'string'));
if HorTotalPixs==0
    error('HorTotalPixs null');
end
%%%%%垂直分辨率-----------------
VerPixles=str2num(get(handles.edit_VerPixles,'string'));
if VerPixles==0   
    error('VerPixles null');
end
%%%%%水平分辨率----------------
HorPixles=str2num(get(handles.edit_HorPixles,'string'));
if HorPixles==0
    error('HorPixles null');
end
if HorPixles>HorTotalPixs %总的水平像素点数 要大于 水平分辨率 才行！
    error('HorPixles<HorTotalPixs');
end
grayscale=str2num(get(handles.edit_grayscale,'string'));%灰度值
if grayscale==0
    grayscale=100;
end
offset=str2num(get(handles.edit_offset,'string'));%像素点的偏移量


TotalNeedSamples=floor(((HorTotalPixs/Pixclock)*VerPixles)/XDelta); %取一帧所需的采样点数
if TotalNeedSamples<N           %resample(X,P,Q)=X*(P/Q)    P*Q要小于intmax('int32')即 2147483647
    %对信号从新采样成点频取样
    %temp1=resample(SampleData(1:TotalNeedSamples),HorPixles*VerPixles,TotalNeedSamples);%方法1:会出现无法取整，没法下采样
    temp1=resample(SampleData(1+offset:TotalNeedSamples+offset),floor(roundn(HorPixles*VerPixles/TotalNeedSamples,-4)*1e4),1e4);%方法2
    if length(temp1)< HorPixles*VerPixles  %长度若不足则补齐
        temp1(length(temp1):HorPixles*VerPixles)=0;
    end
    b=reshape(temp1(1:HorPixles*VerPixles),HorPixles,VerPixles);%把点频信号存成矩阵图片信号
    figure('Name','复现效果');
    imshow(abs(b')*grayscale);%画图片
else
    error('采样点数少于复现所需点数');
end




function edit_grayscale_Callback(hObject, eventdata, handles)
% hObject    handle to edit_grayscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_grayscale as text
%        str2double(get(hObject,'String')) returns contents of edit_grayscale as a double


% --- Executes during object creation, after setting all properties.
function edit_grayscale_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_grayscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_file_Callback(hObject, eventdata, handles)
% hObject    handle to edit_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_file as text
%        str2double(get(hObject,'String')) returns contents of edit_file as a double


% --- Executes during object creation, after setting all properties.
function edit_file_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_offset_Callback(hObject, eventdata, handles)
% hObject    handle to edit_offset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_offset as text
%        str2double(get(hObject,'String')) returns contents of edit_offset as a double


% --- Executes during object creation, after setting all properties.
function edit_offset_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_offset (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
