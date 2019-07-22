function [yPredict] = predict(prob)
     %return the output label based on output of sigmoid(h(x)) value
     if prob >=0.5
        yPredict = 1;
     else
        yPredict = 0;
     end
     
end