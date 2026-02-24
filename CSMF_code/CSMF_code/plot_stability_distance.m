% plot the stability values
function plot_stability_distance(stability_top_record,minK,maxK,positions,fig_width,fig_height)
if ~exist('fig_width', 'var') || isempty(fig_width)
    fig_width = 700;
end
if ~exist('fig_height', 'var') || isempty(fig_height)
    fig_height = 150;
end

N = length(stability_top_record);
nrow = ceil(N/2);
fig_height = fig_height*N;
colors = generateColors(4);
terms = cell(1,maxK-minK+1);
for i = 1:maxK-minK+1
    terms{1,i} = num2str(minK+i-1);
end
figure('position', [600, 200, fig_width, fig_height])
for i = 1:N
    subplot(nrow,N,i)
    stability = stability_top_record{1,i}; stability = log(stability);
    for j = 1:4
        plot(1:maxK-minK+1,stability(j,:),'-o','LineWidth',1,'Color',colors(j,:));
        hold on;
    end
    hold on;
    line([positions(i),positions(i)],get(gca,'Ylim'),'LineStyle','--','LineWidth',1,'Color',[0.6 0.6 0.6])
    legend({'Top 10', 'Top 20', 'Top 30', 'Mean'},'Location','best')
    ax=gca;ax.LineWidth=1;
    set(gca,'FontName','Arail','FontSize',8)
    title(['Data ',num2str(i)],'FontName','Arial','FontSize',12)
    set(gca,'xtick',1:maxK-minK+1);
    set(gca,'xticklabel',terms,'FontName','Arail','FontSize',8);
    xlabel('Ranks','FontName','Arail','FontSize',10)
    ylabel('Stability distance (log)','FontName','Arail','FontSize',10)
end