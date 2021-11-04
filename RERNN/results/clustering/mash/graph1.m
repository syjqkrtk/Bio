x = 10:32;
data = result.VarName1;

figure
plot(x,data,'o--');
xlim([10 32]);
ylim([0.4 0.8]);
xlabel('K-mer size')
ylabel('Accuracy')
legend('MASH, k-medoids')