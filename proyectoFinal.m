function res  = final()
[pregnant,glucose,bloodPressure,triceps,insulin,BMI,DPF,age,class] = textread('pima-indians-diabetes.data.txt' ,'%d%d%d%d%d%f%f%d%d','delimiter', ',');
p=1:768;
%Parametro a considerar como minimo de presion diastolica
lowPressure = 30;
%variables auxiliares
cont = 0;
guardo1 = [];
guardo2 = [];
guardo3 = [];
guardo4 = [];
guardo5 = [];
guardo6 = [];
guardo7 = [];
guardo8 = [];
%cantidad de iteraciones para entrenar los modelos
iteraciones = 350;

%se visualiza los datos para detectar outliers y missing values.

%plot(p, pregnant, 'r');
%plot(p, glucose, 'b');
%plot(p, bloodPressure, 'y-');
%plot(p, triceps, 'b');
%plot(p, insulin, 'r');
%plot(p, BMI, 'b');
%plot(p, DPF, 'r');
%plot(p, age, 'r');
%plot(p, class, 'b');
%title('Entradas y salidas');
%legend('Componente A');
%xlabel('Target No');
%ylabel('Valor Target');
%grid();

%%%%%%%%%%%%%%%%%PRE-PROCESAMIENTO DE DATOS%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Para todos los casos se utiliza la interpolaci?n Neirest-neighbor

%Se interpolan los BMI = 0 en base a los valores vecinos
for i=1:length(BMI)
    if BMI(i) == 0;
        BMI(i) = (BMI(i-1) + BMI(i+1))/2;
        guardo6 = [guardo6 i];
    end
end

%Se interpolan las glucose = 0
for i=1:length(glucose)
    if glucose(i) == 0;
        glucose(i) = (glucose(i-1) + glucose(i+1))/2;
        guardo1 = [guardo1 i];
    end
    
end

%Se considera como umbral mInimo de presion 30,
%Se puede modificar al inicio del c?digo. 
%Referencia http://www.pregmed.org/low-blood-pressure-during-pregnancy.htm
for i=1:length(bloodPressure)
    if bloodPressure(i) <= lowPressure;
        bloodPressure(i) = (bloodPressure(i-1) + bloodPressure(i+1))/2;
       guardo2 = [guardo2 i];
    end
end

%Se revisa cuantos son missing values para el caso de triceps skin fold
%thickness
% De acuerdo a http://ajcn.nutrition.org/content/91/3/635.full se tiene que
% hay muchos missing values 227 = 0
% Se interpolan los datos bas?ndose en los 2 vecinos m?s cercanos hacia
% adelante y atr?s
for i=1:length(triceps)
    if triceps(i) == 0;
        guardo3 = [guardo3 i];
        if((i-2 > 0) && (i+2 < length(triceps)));
            triceps(i) = (triceps(i-2)+triceps(i-1)+ triceps(i+1)+ triceps(i+2))/4;
        end
        if((i-2 < 0) || (i+2 > length(triceps)));
            triceps(i) = (triceps(i-1)+ triceps(i+1))/2;
        end
        %guardo = [guardo i];
    end
end


%Las mujeres embarazadas son mas propensas a ser resistentes a la insulina.
%Existen 374 datos con insulin = 0, sin embargo es un buen indicador de
%diabetes por lo que se realizaron modelos que contemplen esta variable y
%otros que no.
%Para considerar la entrada en el modelo se rellenan los missing values con
%un valor que consiste en la media +- un numero random entre 1 y 25.
for i=1:length(insulin)
    if insulin(i) == 0;
        guardo4 = [guardo4 i];
        desviation = randi(25);
        sign = rand();
        if(sign > 0.5);
            insulin(i) = mean(insulin) + desviation;
        end
        if(sign <= 0.5);
            insulin(i) = mean(insulin) - desviation;
        end
    end
    
end

%DPF no posee outliers. Se mantiene igual.
for i=1:length(DPF)
    if DPF(i) == 0;
        guardo5 = [guardo5 i];
        %bloodPressure(i) = (bloodPressure(i-1) + bloodPressure(i+1))/2;
        %guardo = [guardo i];
    end
end

%La edad se discrimina al inicio del estudio, por lo que esta dentro de los
%rangos normales. Se mantiene igual.
for i=1:length(age)
    if age(i) == 0;
        guardo7 = [guardo7 i];
        %bloodPressure(i) = (bloodPressure(i-1) + bloodPressure(i+1))/2;
        %guardo = [guardo i];
    end
end

matrizJunta = horzcat(pregnant,glucose,bloodPressure,triceps,insulin,BMI,DPF,age,class);
matrizOrdenada = sortrows(matrizJunta,9);

%se eliminan datos de la clase 0 para balancear las clases. 

matrizOrdenada2 = matrizOrdenada([233:768],:);
%matrizOrdenada2 = matrizOrdenada([1:268,500:768],:);



%matrizOrdenada2 es una matriz con las clases balanceadas
matrizTotal = matrizOrdenada2(randperm(536),:);%%%%%%%%%%%%%%%%%%%%%%%Modelo 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Se normalizan los datos, para probar modelos
%matrizTotal(:,1) = matrizTotal(:,1)/norm(matrizTotal(:,1));
%matrizTotal(:,2) = matrizTotal(:,2)/norm(matrizTotal(:,2));
%matrizTotal(:,3) = matrizTotal(:,3)/norm(matrizTotal(:,3));
%matrizTotal(:,4) = matrizTotal(:,4)/norm(matrizTotal(:,4));
%matrizTotal(:,6) = matrizTotal(:,6)/norm(matrizTotal(:,6));
%matrizTotal(:,7) = matrizTotal(:,7)/norm(matrizTotal(:,7));
%matrizTotal(:,8) = matrizTotal(:,8)/norm(matrizTotal(:,8));

%%%%%%%%%%%%%%%%%%%%%Modelo 0%%%%%%%%%%%%%%%%%%%%%%%%%%%
%este modelo se realiza para tener otro punto de comparaci?n

train0 = 1:374;
validation0 = 375:536;
sizeValidation0 = 162;

matrizTrain0 = [matrizTotal([train0],1)';matrizTotal([train0],2)';matrizTotal([train0],3)';matrizTotal([train0],4)';matrizTotal([train0],5)';matrizTotal([train0],6)';matrizTotal([train0],7)';matrizTotal([train0],8)'];
matrizValidation0 = [matrizTotal([validation0],1)';matrizTotal([validation0],2)';matrizTotal([validation0],3)';matrizTotal([validation0],4)';matrizTotal([validation0],5)';matrizTotal([validation0],6)';matrizTotal([validation0],7)';matrizTotal([validation0],8)'];
classTrain0 = matrizTotal([train0],9)';
classValidation0 = matrizTotal([validation0],9)';

%net0 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net0 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net0.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net0.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net0 = train(net0, matrizTrain0 , classTrain0);
y0 = sim(net0, matrizValidation0);


% Calcular RMS y RSD
IA0 = calculateIA(y0, classValidation0,sizeValidation0);  %Debe ser cercano a 1
RMS0 = calculateRMS(y0, classValidation0,sizeValidation0); %Debe ser cercano a 0
RSD0 = calculateRSD(y0, classValidation0,sizeValidation0); %Debe ser cercano a 0


%%%%%%%%%%%%%%%%%%%%%%VALIDACION CRUZADA%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Se utiliza validaci?n cruzada con 10 particiones

%%%%%%%%%%%%%%%%%%%%%Modelo 1%%%%%%%%%%%%%%%%%%%%%%%%%%%

train1 = 1:375;
validation1 = 376:536;
sizeValidation1 = 161;

matrizTrain1 = [matrizTotal([train1],1)';matrizTotal([train1],2)';matrizTotal([train1],3)';matrizTotal([train1],4)';matrizTotal([train1],5)';matrizTotal([train1],6)';matrizTotal([train1],7)';matrizTotal([train1],8)'];
matrizValidation1 = [matrizTotal([validation1],1)';matrizTotal([validation1],2)';matrizTotal([validation1],3)';matrizTotal([validation1],4)';matrizTotal([validation1],5)';matrizTotal([validation1],6)';matrizTotal([validation1],7)';matrizTotal([validation1],8)'];
classTrain1 = matrizTotal([train1],9)';
classValidation1 = matrizTotal([validation1],9)';

%net1 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net1 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net1.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net1.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net1 = train(net1, matrizTrain1 , classTrain1);
y1 = sim(net1, matrizValidation1);


% Calcular RMS y RSD
IA1 = calculateIA(y1, classValidation1,sizeValidation1);  %Debe ser cercano a 1
RMS1 = calculateRMS(y1, classValidation1,sizeValidation1); %Debe ser cercano a 0
RSD1 = calculateRSD(y1, classValidation1,sizeValidation1); %Debe ser cercano a 0


%figure, plotconfusion(classValidation1,y1);
%title('Validaci?n modelo 1');
%figure, plotroc(classValidation1,y1,'ROC modelo 1');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train2 = 55:428;
validation21 = 1:54;
validation22 = 429:536;
sizeValidation2 = 162;

matrizTrain2 = [matrizTotal([train2],1)';matrizTotal([train2],2)';matrizTotal([train2],3)';matrizTotal([train2],4)';matrizTotal([train2],5)';matrizTotal([train2],6)';matrizTotal([train2],7)';matrizTotal([train2],8)'];
matrizValidation2 = [matrizTotal([validation21,validation22],1)';matrizTotal([validation21,validation22],2)';matrizTotal([validation21,validation22],3)';matrizTotal([validation21,validation22],4)';matrizTotal([validation21,validation22],5)';matrizTotal([validation21,validation22],6)';matrizTotal([validation21,validation22],7)';matrizTotal([validation21,validation22],8)'];
classTrain2 = matrizTotal([train2],9)';
classValidation2 = matrizTotal([validation21,validation22],9)';

%net2 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net2 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net2.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net2.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net2 = train(net2, matrizTrain2 , classTrain2);
y2 = sim(net2, matrizValidation2);


% Calcular RMS y RSD
IA2 = calculateIA(y2, classValidation2,sizeValidation2);  %Debe ser cercano a 1
RMS2 = calculateRMS(y2, classValidation2,sizeValidation2); %Debe ser cercano a 0
RSD2 = calculateRSD(y2, classValidation2,sizeValidation2); %Debe ser cercano a 0


%figure, plotconfusion(classValidation2,y2);
%title('Validaci?n modelo 2');
%figure, plotroc(classValidation2,y2,'r',classValidation1,y1,'m');
%figure, plotroc(classValidation1,y1);
%hold on;
%plotroc(classValidation2,y2);

%%%%%%%%%%%%%%%%%%%%%%%Modelo 3%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train3 = 99:473;
validation31 = 1:98;
validation32 = 474:536;
sizeValidation3 = 161;

matrizTrain3 = [matrizTotal([train3],1)';matrizTotal([train3],2)';matrizTotal([train3],3)';matrizTotal([train3],4)';matrizTotal([train3],5)';matrizTotal([train3],6)';matrizTotal([train3],7)';matrizTotal([train3],8)'];
matrizValidation3 = [matrizTotal([validation31,validation32],1)';matrizTotal([validation31,validation32],2)';matrizTotal([validation31,validation32],3)';matrizTotal([validation31,validation32],4)';matrizTotal([validation31,validation32],5)';matrizTotal([validation31,validation32],6)';matrizTotal([validation31,validation32],7)';matrizTotal([validation31,validation32],8)'];
classTrain3 = matrizTotal([train3],9)';
classValidation3 = matrizTotal([validation31,validation32],9)';

%net3 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net3 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net3.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net3.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net3 = train(net3, matrizTrain3 , classTrain3);
y3 = sim(net3, matrizValidation3);


% Calcular RMS y RSD
IA3 = calculateIA(y3, classValidation3,sizeValidation3);  %Debe ser cercano a 1
RMS3 = calculateRMS(y3, classValidation3,sizeValidation3); %Debe ser cercano a 0
RSD3 = calculateRSD(y3, classValidation3,sizeValidation3); %Debe ser cercano a 0


%figure, plotconfusion(classValidation3,y3);
%title('Validaci?n modelo 3');
%figure, plotroc(classValidation3,y3,'ROC modelo 3');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 4%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train4 = 162:536;
validation4 = 1:161;
sizeValidation4 = 161;

matrizTrain4 = [matrizTotal([train4],1)';matrizTotal([train4],2)';matrizTotal([train4],3)';matrizTotal([train4],4)';matrizTotal([train4],5)';matrizTotal([train4],6)';matrizTotal([train4],7)';matrizTotal([train4],8)'];
matrizValidation4 = [matrizTotal([validation4],1)';matrizTotal([validation4],2)';matrizTotal([validation4],3)';matrizTotal([validation4],4)';matrizTotal([validation4],5)';matrizTotal([validation4],6)';matrizTotal([validation4],7)';matrizTotal([validation4],8)'];
classTrain4 = matrizTotal([train4],9)';
classValidation4 = matrizTotal([validation4],9)';

%net4 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net4 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net4.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net4.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net4 = train(net4, matrizTrain4 , classTrain4);
y4 = sim(net4, matrizValidation4);


% Calcular RMS y RSD
IA4 = calculateIA(y4, classValidation4,sizeValidation4);  %Debe ser cercano a 1
RMS4 = calculateRMS(y4, classValidation4,sizeValidation4); %Debe ser cercano a 0
RSD4 = calculateRSD(y4, classValidation4,sizeValidation4); %Debe ser cercano a 0


%figure, plotconfusion(classValidation4,y4);
%title('Validaci?n modelo 4');
%figure, plotroc(classValidation4,y4,'ROC modelo 4');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 5%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train51 = 1:54;
train52 = 216:536;
validation5 = 55:215;
sizeValidation5 = 161;

matrizTrain5 = [matrizTotal([train51,train52],1)';matrizTotal([train51,train52],2)';matrizTotal([train51,train52],3)';matrizTotal([train51,train52],4)';matrizTotal([train51,train52],5)';matrizTotal([train51,train52],6)';matrizTotal([train51,train52],7)';matrizTotal([train51,train52],8)'];
matrizValidation5 = [matrizTotal([validation5],1)';matrizTotal([validation5],2)';matrizTotal([validation5],3)';matrizTotal([validation5],4)';matrizTotal([validation5],5)';matrizTotal([validation5],6)';matrizTotal([validation5],7)';matrizTotal([validation5],8)'];
classTrain5 = matrizTotal([train51,train52],9)';
classValidation5 = matrizTotal([validation5],9)';

%net5 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net5 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net5.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net5.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net5 = train(net5, matrizTrain5 , classTrain5);
y5 = sim(net5, matrizValidation5);

% Calcular RMS y RSD
IA5 = calculateIA(y5, classValidation5,sizeValidation5);  %Debe ser cercano a 1
RMS5 = calculateRMS(y5, classValidation5,sizeValidation5); %Debe ser cercano a 0
RSD5 = calculateRSD(y5, classValidation5,sizeValidation5); %Debe ser cercano a 0


%figure, plotconfusion(classValidation5,y5);
%title('Validaci?n modelo 5');
%figure, plotroc(classValidation5,y5,'ROC modelo 5');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 6%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train61 = 1:98;
train62 = 260:536;
validation6 = 99:259;
sizeValidation6 = 161;

matrizTrain6 = [matrizTotal([train61,train62],1)';matrizTotal([train61,train62],2)';matrizTotal([train61,train62],3)';matrizTotal([train61,train62],4)';matrizTotal([train61,train62],5)';matrizTotal([train61,train62],6)';matrizTotal([train61,train62],7)';matrizTotal([train61,train62],8)'];
matrizValidation6 = [matrizTotal([validation6],1)';matrizTotal([validation6],2)';matrizTotal([validation6],3)';matrizTotal([validation6],4)';matrizTotal([validation6],5)';matrizTotal([validation6],6)';matrizTotal([validation6],7)';matrizTotal([validation6],8)'];
classTrain6 = matrizTotal([train61,train62],9)';
classValidation6 = matrizTotal([validation6],9)';

%net6 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net6 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net6.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net6.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net6 = train(net6, matrizTrain6 , classTrain6);
y6 = sim(net6, matrizValidation6);

% Calcular RMS y RSD
IA6 = calculateIA(y6, classValidation6,sizeValidation6);  %Debe ser cercano a 1
RMS6 = calculateRMS(y6, classValidation6,sizeValidation6); %Debe ser cercano a 0
RSD6 = calculateRSD(y6, classValidation6,sizeValidation6); %Debe ser cercano a 0


%figure, plotconfusion(classValidation6,y6);
%title('Validaci?n modelo 6');
%figure, plotroc(classValidation6,y6,'ROC modelo 6');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 7%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train71 = 1:162;
train72 = 325:536;
validation7 = 163:324;
sizeValidation7 = 162;

matrizTrain7 = [matrizTotal([train71,train72],1)';matrizTotal([train71,train72],2)';matrizTotal([train71,train72],3)';matrizTotal([train71,train72],4)';matrizTotal([train71,train72],5)';matrizTotal([train71,train72],6)';matrizTotal([train71,train72],7)';matrizTotal([train71,train72],8)'];
matrizValidation7 = [matrizTotal([validation7],1)';matrizTotal([validation7],2)';matrizTotal([validation7],3)';matrizTotal([validation7],4)';matrizTotal([validation7],5)';matrizTotal([validation7],6)';matrizTotal([validation7],7)';matrizTotal([validation7],8)'];
classTrain7 = matrizTotal([train71,train72],9)';
classValidation7 = matrizTotal([validation7],9)';

%net7 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net7 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net7.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net7.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net7 = train(net7, matrizTrain7 , classTrain7);
y7 = sim(net7, matrizValidation7);

% Calcular RMS y RSD
IA7 = calculateIA(y7, classValidation7,sizeValidation7);  %Debe ser cercano a 1
RMS7 = calculateRMS(y7, classValidation7,sizeValidation7); %Debe ser cercano a 0
RSD7 = calculateRSD(y7, classValidation7,sizeValidation7); %Debe ser cercano a 0


%figure, plotconfusion(classValidation7,y7);
%title('Validaci?n modelo 7');
%figure, plotroc(classValidation7,y7,'ROC modelo 7');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 8%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train81 = 1:216;
train82 = 378:536;
validation8 = 217:377;
sizeValidation8 = 161;

matrizTrain8 = [matrizTotal([train81,train82],1)';matrizTotal([train81,train82],2)';matrizTotal([train81,train82],3)';matrizTotal([train81,train82],4)';matrizTotal([train81,train82],5)';matrizTotal([train81,train82],6)';matrizTotal([train81,train82],7)';matrizTotal([train81,train82],8)'];
matrizValidation8 = [matrizTotal([validation8],1)';matrizTotal([validation8],2)';matrizTotal([validation8],3)';matrizTotal([validation8],4)';matrizTotal([validation8],5)';matrizTotal([validation8],6)';matrizTotal([validation8],7)';matrizTotal([validation8],8)'];
classTrain8 = matrizTotal([train81,train82],9)';
classValidation8 = matrizTotal([validation8],9)';

%net8 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net8 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net8.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net8.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net8 = train(net8, matrizTrain8 , classTrain8);
y8 = sim(net8, matrizValidation8);

% Calcular RMS y RSD
IA8 = calculateIA(y8, classValidation8,sizeValidation8);  %Debe ser cercano a 1
RMS8 = calculateRMS(y8, classValidation8,sizeValidation8); %Debe ser cercano a 0
RSD8 = calculateRSD(y8, classValidation8,sizeValidation8); %Debe ser cercano a 0

%figure, plotconfusion(classValidation8,y8);
%title('Validaci?n modelo 8');
%figure, plotroc(classValidation8,y8,'ROC modelo 8');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 9%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train91 = 1:269;
train92 = 431:536;
validation9 = 270:430;
sizeValidation9 = 161;

matrizTrain9 = [matrizTotal([train91,train92],1)';matrizTotal([train91,train92],2)';matrizTotal([train91,train92],3)';matrizTotal([train91,train92],4)';matrizTotal([train91,train92],5)';matrizTotal([train91,train92],6)';matrizTotal([train91,train92],7)';matrizTotal([train91,train92],8)'];
matrizValidation9 = [matrizTotal([validation9],1)';matrizTotal([validation9],2)';matrizTotal([validation9],3)';matrizTotal([validation9],4)';matrizTotal([validation9],5)';matrizTotal([validation9],6)';matrizTotal([validation9],7)';matrizTotal([validation9],8)'];
classTrain9 = matrizTotal([train91,train92],9)';
classValidation9 = matrizTotal([validation8],9)';

%net9 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net9 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net9.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net9.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net9 = train(net9, matrizTrain9 , classTrain9);
y9 = sim(net9, matrizValidation9);

% Calcular RMS y RSD
IA9 = calculateIA(y9, classValidation9, sizeValidation9);  %Debe ser cercano a 1
RMS9 = calculateRMS(y9, classValidation9,sizeValidation9); %Debe ser cercano a 0
RSD9 = calculateRSD(y9, classValidation9,sizeValidation9); %Debe ser cercano a 0

%figure, plotconfusion(classValidation9,y9);
%title('Validaci?n modelo 9');
%figure, plotroc(classValidation9,y9,'ROC modelo 9');

%%%%%%%%%%%%%%%%%%%%%%%Modelo 10%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train101 = 1:322;
train102 = 484:536;
validation10 = 323:483;
sizeValidation10 = 161;

matrizTrain10 = [matrizTotal([train101,train102],1)';matrizTotal([train101,train102],2)';matrizTotal([train101,train102],3)';matrizTotal([train101,train102],4)';matrizTotal([train101,train102],5)';matrizTotal([train101,train102],6)';matrizTotal([train101,train102],7)';matrizTotal([train101,train102],8)'];
matrizValidation10 = [matrizTotal([validation10],1)';matrizTotal([validation10],2)';matrizTotal([validation10],3)';matrizTotal([validation10],4)';matrizTotal([validation10],5)';matrizTotal([validation10],6)';matrizTotal([validation10],7)';matrizTotal([validation10],8)'];
classTrain10 = matrizTotal([train101,train102],9)';
classValidation10 = matrizTotal([validation10],9)';

%net10 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[8 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
%con 16 neuronas en capa oculta
net10 = newff([minmax(matrizTotal([1:536],1)');minmax(matrizTotal([1:536],2)');minmax(matrizTotal([1:536],3)');minmax(matrizTotal([1:536],4)');minmax(matrizTotal([1:536],5)');minmax(matrizTotal([1:536],6)');minmax(matrizTotal([1:536],7)');minmax(matrizTotal([1:536],8)')],[16 1], {'tansig', 'purelin'}, 'trainscg', 'learngd', 'mse' );
net10.trainParam.epochs = iteraciones; %Cantidad m?xima de iteraciones
net10.trainParam.show = 50; %Cada cuantas iteraciones ver los resultados parciales

%red.trainParam.goal= ; %Error m?ximo a encontrar para detener el entrenamiento
net10 = train(net10, matrizTrain10 , classTrain10);
y10 = sim(net10, matrizValidation10);
length(classValidation10);
length(y10);
% Calcular RMS y RSD
IA10 = calculateIA(y10, classValidation10, sizeValidation10);  %Debe ser cercano a 1
RMS10 = calculateRMS(y10, classValidation10,sizeValidation10); %Debe ser cercano a 0
RSD10 = calculateRSD(y10, classValidation10,sizeValidation10); %Debe ser cercano a 0


%figure, plotconfusion(classValidation10,y10);
%title('Validaci?n modelo 10');
figure, plotconfusion(classValidation0,y0,'0',classValidation1,y1,'1',classValidation2,y2,'2',classValidation3,y3,'3',classValidation4,y4,'4',classValidation5,y5,'5',classValidation6,y6,'6',classValidation7,y7,'7',classValidation8,y8,'8 ',classValidation9,y9,'9',classValidation10,y10,'10');
figure, plotroc(classValidation0,y0,'Modelo 0',classValidation1,y1,'Modelo 1',classValidation2,y2,'Modelo 2',classValidation3,y3,'Modelo 3',classValidation4,y4,'Modelo 4',classValidation5,y5,'Modelo 5',classValidation6,y6,'Modelo 6',classValidation7,y7,'Modelo 7',classValidation8,y8,'Modelo 8 ',classValidation9,y9,'Modelo 9',classValidation10,y10,'Modelo 10');

IA =  [];
IA = [IA IA0];
IA = [IA IA1];
IA = [IA IA2];
IA = [IA IA3];
IA = [IA IA4];
IA = [IA IA5];
IA = [IA IA6];
IA = [IA IA7];
IA = [IA IA8];
IA = [IA IA9];
IA = [IA IA10];
IA

RMS = [];
RMS = [RMS RMS0];
RMS = [RMS RMS1];
RMS = [RMS RMS2];
RMS = [RMS RMS3];
RMS = [RMS RMS4];
RMS = [RMS RMS5];
RMS = [RMS RMS6];
RMS = [RMS RMS7];
RMS = [RMS RMS8];
RMS = [RMS RMS9];
RMS = [RMS RMS10];
RMS

RSD = [];
RSD = [RSD RSD0];
RSD = [RSD RSD1];
RSD = [RSD RSD2];
RSD = [RSD RSD3];
RSD = [RSD RSD4];
RSD = [RSD RSD5];
RSD = [RSD RSD6];
RSD = [RSD RSD7];
RSD = [RSD RSD8];
RSD = [RSD RSD9];
RSD = [RSD RSD10];
RSD

%No se considera pregnant
MissingValues = [];
MissingValues = [MissingValues length(guardo1)];
MissingValues = [MissingValues length(guardo2)];
MissingValues = [MissingValues length(guardo3)];
MissingValues = [MissingValues length(guardo4)];
MissingValues = [MissingValues length(guardo5)];
MissingValues = [MissingValues length(guardo6)];
MissingValues = [MissingValues length(guardo7)];
MissingValues

minmax(matrizTotal([1:536],1)')
minmax(matrizTotal([1:536],2)')
minmax(matrizTotal([1:536],3)')
minmax(matrizTotal([1:536],4)')
minmax(matrizTotal([1:536],5)')
minmax(matrizTotal([1:536],6)')
minmax(matrizTotal([1:536],7)')
minmax(matrizTotal([1:536],8)')

end

%%%%%%%%%%%%%%%%%%%%%%%Indices%%%%%%%%%%%%%
function IA = calculateIA(y, yValidation, size)
    m=size;
    num = 0;
    den = 0;
    for i=1:m
        num = num + (y(i)-yValidation(i))^2;
        den = den + (abs(y(i))+abs(yValidation(i)))^2;
    end
    IA = 1 - num/den;  %Debe ser cercano a 1
    
end

function RMS = calculateRMS(y, yValidation, size)
    m=size;
    num = 0;
    den = 0;
    for i=1:m
        num = num + (y(i)-yValidation(i))^2;
        den = den + (abs(y(i))+abs(yValidation(i)))^2;
    end
    RMS = sqrt(num/den);
    
end

function RSD = calculateRSD(y, yValidation, size)
    m=size;
    num = 0;
    den = 0;
    for i=1:m
        num = num + (y(i)-yValidation(i))^2;
        den = den + (abs(y(i))+abs(yValidation(i)))^2;
    end
    RSD = sqrt(num/m) %Debe ser cercano a 0
    
end



