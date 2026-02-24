function plot_heatmap(W,vecPara,varargin)
% Default configuration
Transp = 'F';
Samplename = [];
% Read optional parameters
if (rem(length(varargin),2) == 1)
    error('Optional parameters should always go by pairs');
else
    for i = 1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'TRANSP',      Transp = varargin{i+1};
            case 'SAMPLENAME',  Samplename = varargin{i+1};
            otherwise
                error(['Unrecognized option: ',varargin{i}]);
        end
    end
end

S = sum(vecPara);
vecN = zeros(1,length(vecPara)-1);
for i = 1:length(vecPara)-1
    s = 0;
    for j = 1:i
        s = s + vecPara(j);
    end
    vecN(i) = s;
end
terms = cell(1,vecPara(1));
for k = 1:vecPara(1)
    terms{1,k} = ['C-',num2str(k)];
end
for i = 2:length(vecPara)
    termi = cell(1,vecPara(i));
    for j = 1:vecPara(i)
        termi{1,j} = ['S',num2str(i-1),'-',num2str(j)];
    end
    terms = [terms,termi];
end
figure;
if isequal(Transp,'F')
    imagesc(W)
    hold on;
    for t = 1:length(vecPara)-1
        line([vecN(t)+0.5,vecN(t)+0.5],[0,size(W,1)],'LineStyle','--','LineWidth',1,'Color','red')
        hold on;
    end
    set(gca,'xtick',1:S);
    set(gca,'xticklabel',terms,'FontName','Arail','FontSize',10);
    xtickangle(45)
    xlabel('Latent patterns','FontName','Arail','FontSize',10)
    ylabel('Features','FontName','Arail','FontSize',10)
    title('Basis matrix W','FontName','Arail','FontSize',12)
    set(gca,'FontName','Arail')
    colorbar
else
    H = W';
    imagesc(H)
    hold on;
    for t = 1:length(vecPara)-1
        line([0,size(H,2)],[vecN(t)+0.5,vecN(t)+0.5],'LineStyle','--','LineWidth',1,'Color','red')
        hold on;
    end
    set(gca,'ytick',1:S);
    set(gca,'yticklabel',terms,'FontName','Arail','FontSize',10);
    xtickangle(45)
    if ~isempty(Samplename)
        set(gca,'xtick',1:size(H,2));
        Samplename = strrep(Samplename,'_','-');
        set(gca,'xticklabel',Samplename,'FontName','Arail','FontSize',10);
    end
    ylabel('Latent patterns','FontName','Arail','FontSize',10)
    xlabel('Samples','FontName','Arail','FontSize',10)
    title('Coefficient matrix H','FontName','Arail','FontSize',12)
    set(gca,'FontName','Arail')
    colorbar
end
   

