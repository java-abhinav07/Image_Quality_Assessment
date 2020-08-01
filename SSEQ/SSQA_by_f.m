function score = SSQA_by_f( feature)
    warning ('off', 'Octave:data-file-in-path')

    load model;
    pkg load image statistics optim signal;
    addpath('/home/anil/Desktop/Dehazing_Abhinav/libsvm/matlab');
    feature =setscale(feature,fmax,fmin);
    [pred_class, acc, p] = svmpredict(ones(1,1), feature, svmmodel,'-b 1 -q');
    q=zeros(1,5);
    for j=1:5
        [q(:,j), reg_acc, dec] = svmpredict(ones(1,1), feature, svrmodel{j}, '-q');
    end
    Q=sum(p.*q , 2);
    score=Q*50+50;
end
