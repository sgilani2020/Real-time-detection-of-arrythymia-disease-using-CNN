
%Checking Unprocessed Id in server
function [UnProcessedIds] = CheckUnProcessedIds()
 
UnProcessedIds = struct();
WebAddress = 'http://cardiaccare.net/apis/get_rawids.php';
options = weboptions('Timeout',Inf);
Response = webwrite(WebAddress,'rid',0,options);
SuccessId = strfind(Response,'success');
UnProcessedIds.WebReadSuccess = str2double(Response(SuccessId+9));
if(UnProcessedIds.WebReadSuccess==1)
    responsearray = Response(strfind(Response,'[')+1:strfind(Response,']')-1);
    eachrecord = strsplit(responsearray,',');
    for i=1:length(eachrecord)
        casestr = cell2mat(eachrecord(i));
        commasind = strfind(casestr,'"');
        UnProcessedIds.Records(i)=str2double(casestr(commasind(end-1)+1:commasind(end)-1));
    end
end
 
end

%CNN Classifier 
clc
clear
close all
disp('Started Classifer')
DataStorePath = 'E:\GIKi\ECG\DataSetImages';
ImgsDataStore = imageDatastore(DataStorePath, 'IncludeSubfolders',true, 'LabelSource','foldernames');
[TrainingImgs, TestImgs] = splitEachLabel(ImgsDataStore, 0.7, 'randomized');
% numClasses = numel(categories(ImgsDataStore.Labels));
% disp('Data Store Created')
% DeepNet = alexnet;
% NetLayers = DeepNet.Layers;
% NetLayers(end-2) = fullyConnectedLayer(numClasses);
% NetLayers(end) = classificationLayer;
% disp('Network Created')
% TrainingOptions = trainingOptions('sgdm', 'InitialLearnRate',0.001, 'MiniBatchSize',512, 'MaxEpochs',5, 'ExecutionEnvironment','parallel', 'OutputFcn',@plotTrainingAccuracy);
% 
% [TrainedNetwork, Info] = trainNetwork(TrainingImgs, NetLayers, TrainingOptions);
% disp('Network Trained')
load MyNetwork
[Predictions, Scores] = classify(TrainedNetwork, TestImgs);
disp('Classification Done')
%numequal = nnz(TestImgs.Labels, Predictions);
[ConfMat, PredLabels] = confusionmat(TestImgs.Labels, Predictions);
disp('Done')
%Counting Beats
function [NoBeats] = CountBeats(Beats)
 
NoBeats = struct();
NoBeats.AAPB = count(Beats',"AAPB");
NoBeats.AEB = count(Beats',"AEB");
NoBeats.APB = count(Beats',"APB");
NoBeats.FPNB = count(Beats',"FPNB");
NoBeats.FVNB = count(Beats',"FVNB");
NoBeats.LBBBB = count(Beats',"LBBBB");
NoBeats.NB = count(Beats',"NB");
NoBeats.NEB = count(Beats',"NEB");
NoBeats.NPB = count(Beats',"NPB");
NoBeats.PB = count(Beats',"PB");
NoBeats.PVC = count(Beats',"PVC");
NoBeats.RBBBB = count(Beats',"RBBBB");
NoBeats.SPB = count(Beats',"SPB");
NoBeats.UB = count(Beats',"UB");
NoBeats.VEB = count(Beats',"VEB");
NoBeats.Total=NoBeats.AAPB+NoBeats.AEB+NoBeats.APB+NoBeats.FPNB+NoBeats.FVNB+NoBeats.LBBBB+NoBeats.NB+NoBeats.NEB+NoBeats.NPB+NoBeats.PB+NoBeats.PVC+NoBeats.RBBBB+NoBeats.SPB+NoBeats.UB+NoBeats.VEB;
end
%Creating Report
function [Report] = CreateReport(RawECGValues)
 
Report = struct(); 
load Filter
load MyNetwork
Report.FilteredECG = BP_Filter(RawECGValues);
[p,s,mu] = polyfit((1:numel(Report.FilteredECG))',Report.FilteredECG,20);
f_y = polyval(p,(1:numel(Report.FilteredECG))',[],mu);
DetrendedECG = Report.FilteredECG - f_y;
Report.SmoothECG = 1.7.*smoothdata(DetrendedECG,'gaussian',20)./(sum(abs(DetrendedECG))/length(DetrendedECG));
[~,PosPeakloc] = findpeaks(Report.SmoothECG,'MinPeakHeight',3,'MinPeakDistance',150);
[~,NegPeakloc] = findpeaks(-Report.SmoothECG,'MinPeakHeight',3,'MinPeakDistance',150);
SortedLocs = sort([PosPeakloc;NegPeakloc]);
in = 1;
for i=1:length(SortedLocs)-1
    if(SortedLocs(i+1)-SortedLocs(i)>150)
        IndLocs(in) = SortedLocs(i);
        in = in+1;
    end
end
IndLocs = [IndLocs SortedLocs(end)];
for i=1:length(IndLocs)-1
    SingleBeatValues = Report.SmoothECG(IndLocs(i)-70:IndLocs(i)+145);
    Report.SBeat(i) = {SingleBeatValues};
    fig=figure;
    fig.Visible='off';
    plot(SingleBeatValues)
    axis off
    ImgName=strcat('Input.tiff');
    saveas(fig,ImgName,'tiff');
    figrsz=imresize(imread(ImgName,'tiff'),[227 227]);
    imwrite(figrsz,ImgName,'tiff');
    close all
    I = imread('Input.tiff');
    [pred(i),scor] = classify(TrainedNetwork,I);
end
 
Report.AnnXInd=IndLocs(1:end-1);
Report.BeatPrediction=pred;
end
%ECG Main
clc
clear
close all
% wfdb2mat('mitdb/100')
% save('Filter','BP_Filter');
dataset='106';
load Filter
% load 100m.mat
[Raw_ECG_Signal,Fs,tm]=rdsamp(dataset);
[ann, anntype]=rdann(dataset,'atr',[],108000);
% anntime=wfdbtime(dataset,ann);
Channel1_Raw_ECG=Raw_ECG_Signal(1:108000,1);
Filtered_Channel1=BP_Filter(Channel1_Raw_ECG);
 
[p,s,mu] = polyfit((1:numel(Filtered_Channel1))',Filtered_Channel1,20);
f_y = polyval(p,(1:numel(Filtered_Channel1))',[],mu);
Detrended_Channel1_ECG = Filtered_Channel1 - f_y;        % Detrend data
Smoothed_Channel1=1.7.*smoothdata(Detrended_Channel1_ECG,'gaussian',20);
 
[~,loc]=findpeaks(Smoothed_Channel1,'MinPeakHeight',0.5,'MinPeakDistance',200);
for i=5:length(loc)-1
    pd=Smoothed_Channel1(loc(i)-70:loc(i)+145);
    fig=figure;
    fig.Visible='off';
    plot(pd)
    axis off
    fpname=strcat('E:\GIKi\ECG\Normal\Img',num2str(i),'.tiff');
    saveas(fig,fpname,'tiff')
    figrsz=imresize(imread(fpname,'tiff'),[227 227]);    
    imwrite(figrsz,fpname,'tiff')
end
 
 
% TS = dsp.TimeScope('SampleRate',360,...
%                       'TimeSpan',5,...
%                       'YLimits',[-1 1],...
%                       'ShowGrid',true,...
%                       'NumInputPorts',3,...
%                       'LayoutDimensions',[3 1],...
%                       'TimeAxisLabels','Bottom',...
%                       'Title','Noisy and Filtered Signals');
% tic;
% while toc<5
%     toc
%     TS(Channel1_Raw_ECG, Detrended_Channel1_ECG, Smoothed_Channel1);
% end
% release(TS)
 
 
wt = modwt(Smoothed_Channel1,5);
wtrec = zeros(size(wt));
wtrec(4:5,:) = wt(4:5,:);
y = imodwt(wtrec,'sym4');
y = abs(y).^2;
[~,locs] = findpeaks(y,'MinPeakHeight',0.35,'MinPeakDistance',200);
 
for i=1:length(locs)-1
    hr(i)=60*(360/(locs(i+1)-locs(i)));
end
 
[a,b,c,d,e,f]=rdann(dataset,'atr');
%Get Raw ECG
function [RawECG] = GetRawECG(RecordId)
 
RawECG = struct();
WebAddress = 'http://cardiaccare.net/apis/get_raw.php';
options = weboptions('Timeout',Inf);
Response = webwrite(WebAddress,'rid',RecordId,options);
SuccessId = strfind(Response,'success');
RawECG.WebReadSuccess = str2double(Response(SuccessId+9));
if RawECG.WebReadSuccess == 1
    errstr = erase(Response,"\r\n");
    ind = strfind(errstr,'"');
    val = str2double(strsplit(errstr(ind(end-1)+1:ind(end)-1),','));
    RawECG.Values = val(1:end-1)';
end
end

%HTTP Request for server
clc
clear
close all
 
for cloop=1:1
    UnProcessedIds = CheckUnProcessedIds();
    if UnProcessedIds.WebReadSuccess ~= 1
        continue
    else
        for i=1:length(UnProcessedIds.Records)
            RawECG = GetRawECG(UnProcessedIds.Records(i));
            if RawECG.WebReadSuccess ~= 1
                continue
            else
                Report = CreateReport(RawECG.Values);
                UploadedReport = SendReport(UnProcessedIds.Records(i), Report.SmoothECG, Report.AnnXInd, Report.BeatPrediction);
                if UploadedReport.WebReadSuccess ~=1
                    continue
                end
            end
        end
    end
end
 
%MIT-BIH report

function [Report] = MIT_BIH_Report(PathtoRecord)
 
Report = struct();
load Filter
load MyNetwork
% DataSet = 'database/mitdb/100';
[RawECGSignal,Fs,tm]=rdsamp(PathtoRecord);
% [annIndex, annBeat, annSubtype, annChan, annNum, annRhythm]=rdann(DataSet,'atr');
Report.RawECGValues=RawECGSignal(1:3600,1);
Report.FilteredECG = BP_Filter(Report.RawECGValues);
[p,s,mu] = polyfit((1:numel(Report.FilteredECG))',Report.FilteredECG,20);
f_y = polyval(p,(1:numel(Report.FilteredECG))',[],mu);
DetrendedECG = Report.FilteredECG - f_y;
Report.SmoothECG = 1.7.*smoothdata(DetrendedECG,'gaussian',20)./(sum(abs(DetrendedECG))/length(DetrendedECG));
[~,PosPeakloc] = findpeaks(Report.SmoothECG,'MinPeakHeight',3,'MinPeakDistance',150);
[~,NegPeakloc] = findpeaks(-Report.SmoothECG,'MinPeakHeight',3,'MinPeakDistance',150);
SortedLocs = sort([PosPeakloc;NegPeakloc]);
in = 1;
for i=1:length(SortedLocs)-1
    if(SortedLocs(i+1)-SortedLocs(i)>150)
        IndLocs(in) = SortedLocs(i);
        in = in+1;
    end
end
IndLocs = [IndLocs SortedLocs(end)];
for i=1:length(IndLocs)-1
    SingleBeatValues = Report.SmoothECG(IndLocs(i)-70:IndLocs(i)+145);
    Report.SBeat(i) = {SingleBeatValues};
    fig=figure;
    fig.Visible='off';
    plot(SingleBeatValues)
    axis off
    ImgName=strcat('Input.tiff');
    saveas(fig,ImgName,'tiff');
    figrsz=imresize(imread(ImgName,'tiff'),[227 227]);
    imwrite(figrsz,ImgName,'tiff');
    close all
    I = imread('Input.tiff');
    [pred(i),scor] = classify(TrainedNetwork,I);
end
 
Report.AnnXInd=IndLocs(1:end-1);
Report.BeatPrediction=pred;
 
End
Ploting Accuracy 
function plotTrainingAccuracy(info)
 
persistent plotObj
 
if info.State == "start"
    plotObj = animatedline;
    xlabel("Iteration")
    ylabel("Training Accuracy")
elseif info.State == "iteration"
    addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
    drawnow limitrate nocallbacks
end
 
end

%Sending report back to server
function [UploadedReport] = SendReport(RecordId, ProcessedData, AnnXInd, BeatPrediction)
 
UploadedReport = struct();
WebAddress = 'http://cardiaccare.net/apis/send_p.php';
options = weboptions('Timeout',Inf);
Response = webwrite(WebAddress,'rid',RecordId,'pdata',ProcessedData,'annotationsx',AnnXInd,'annotationsy',BeatPrediction,options);
SuccessId = strfind(Response,'success');
UploadedReport.WebReadSuccess = str2double(Response(SuccessId+9));
end

%testtb
conn = database('fornaxhu_CardicCare','fornaxhu_Umair','NiUsrp2920', 'Vendor','MySQL', 'Server','www.fornaxhub.com');
conn = database('fornaxhu_CardicCare','wptest','Asdf@1234567890', 'Vendor','MySQL', 'Server','20.188.98.39');

 
selectquery = 'SELECT * FROM Patients';
colnames = {'FullName','EmailID','ContactNo','Address'};
data = {"Qasim Gilani","qa***@gmail.com",420440,"Neelum Jhelum"};
data_table = cell2table(data,'VariableNames',colnames);
tablename = 'Patients';
datainsert(conn,tablename,colnames,data_table)
 
data = select(conn,selectquery)
 
% result=runsqlscript(conn,'testQueries.sql')
result=exec(conn,'CREATE TABLE fruits (fruit_name VARCHAR(20) NOT NULL PRIMARY KEY,color VARCHAR(20),price INT)')
result=exec(conn,'DROP TABLE fruits')
result=exec(conn,'SHOW TABLES')
select(conn,selectquery)
result=fetch(result);
result.Data
 
 
 
close(conn)
