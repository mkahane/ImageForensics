%Complete process of compressing an uncompressed image twice
function multiple_compression(filename)
%By Matthew Kahane
%A function to simulate the fourier artifacts in coefficient histograms due
%to multiple compression.
    clc
    close all
    
    [r c] = getUserInput;
    r = str2double(r);
    c = str2double(c);
    c_levels = getCLevels;
    ncompressions = length(c_levels);
    
    uncompressed = load_image(filename);

    %Loop to compress as many times as user indicated
    for compression = 1:ncompressions
        q_table = c_levels(compression).*ones(8);
        [compressed coeffs] = jpeg_compress(uncompressed,q_table,r,c);
        decompressed = jpeg_decompress(compressed,q_table);
        [xcenters xedges nbins] = histParams(coeffs);
        histfun = histc(coeffs,xedges);
        histfun = zeroMean(histfun);
        FT = fft(histfun);
        makePlots(compression,coeffs,xcenters,FT,r,c,c_levels)
        uncompressed = decompressed;
    end
       
end

%Gets user input to find out which DCT coefficient to Analyze
function [r c] = getUserInput
    prompt = {'DCT Coeff. Row: ', 'DCT Coeff. Col'};
    dlg_title = 'Input';
    num_lines = 1;
    def = {'1','1'};
    answer = inputdlg(prompt,dlg_title,num_lines,def);
   
    r = cell2mat(answer(1));
    c = cell2mat(answer(2));
end

%Gets compression levels from user
function c_levels = getCLevels
    x = inputdlg(strcat('enter space separated compression levels')...
        , 'C Levels', [1 50]);
    c_levels = str2num(x{:});
end

%Makes histogram plots and fourier plots of histogram
function makePlots(compression,coeffs,xcenters,FT,r,c,c_levels)
    levels_so_far = c_levels(1:compression);
    figure(1)
    subplot(length(c_levels),1,compression)
    hist(coeffs,xcenters)
    xlabel('Compressed DCT Coefficient Values')
    ylabel('Number of Coefficients With Value')
    title({strcat('Histogram of Compressed DCT Values After Compressions'...
        ,mat2str(levels_so_far)),strcat('For DCT Coefficient (',num2str(r)...
        ,',',num2str(c),')')});
    
    figure(2)
    subplot(length(c_levels),1,compression)
    plot(abs(fftshift(FT)));
    xlabel('spatial frequency')
    ylabel('magnitude of fourier transform')
    title({strcat('Magnitude of Histogram Fourier Transform After Compressions'...
        ,mat2str(levels_so_far)),strcat('For DCT Coefficient (',num2str(r)...
        ,',',num2str(c),')')});
    set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
    set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);

end
function im = load_image(filename)
    im = imread(filename);
    %im = rgb2ycbcr(im);
    im = double(im);
    im = im(:,:,1);%Only one channel necessary
    sz = size(im);
    sz = floor(sz/8)*8;
    im = im(1:sz(1),1:sz(2));%Make sure image can be split into 8x8 blocks
end

%Compresses an image by traversing image and applying DCT to 8x8 blocks.
%Then these blocks are quantized by the input table.  The floor of this is
%taken, and the coefficients are returned as the compressed coefficients.
%The compressed image is then returned for decompression later
function [compressed all_coeffs] = jpeg_compress(uncompressed,q_table,row,col)
    nrows = size(uncompressed,1);
    ncols = size(uncompressed,2);
    compressed = zeros(nrows,ncols);
    all_coeffs = zeros(nrows/8,ncols/8);
    for i = 1:nrows/8
        for j = 1:ncols/8
            topleft_r = (i-1)*8;
            topleft_c = (j-1)*8;
            Block = uncompressed(topleft_r+1:topleft_r+8,topleft_c+1:topleft_c+8);
            DBlock = dct2(Block);
            quantized = floor(DBlock./q_table);
            compressed(topleft_r+1:topleft_r+8,topleft_c+1:topleft_c+8)...
                = quantized;
            all_coeffs(i,j) = quantized(row,col);
        end
    end
    all_coeffs = reshape(all_coeffs,(nrows/8)*(ncols/8),1);
end

%Decompresses an image in a similar way to compression using iDCT on each
%blcok
function decompressed = jpeg_decompress(compressed, q_table)
    nrows = size(compressed,1);
    ncols = size(compressed,2);
    decompressed = zeros(nrows,ncols);
    for i = 1:nrows/8
        for j = 1:ncols/8
            topleft_r = (i-1)*8;
            topleft_c = (j-1)*8;
            DBlock = compressed(topleft_r+1:topleft_r+8,topleft_c+1:topleft_c+8);
            DBlock = DBlock.*q_table;
            Block = idct2(DBlock);
            decompressed(topleft_r+1:topleft_r+8,topleft_c+1:topleft_c+8)...
                = Block;
        end
    end
end

%Explicitly gets parameters for histogram
function [xcenters xedges nbins] = histParams(data)
    minimum = min(data);
    maximum = max(data);
    binsize = 1;
    xcenters = minimum:binsize:maximum;
    xedges = xcenters - binsize/2;
    nbins = length(xcenters);
end

%To make sure low frequencies do not dominate histogram plot
function zMeaned = zeroMean(data)
    sig = mean(data);
    zMeaned = data-sig;
end