# 图像处理综合实验

* 无 36
* 李思涵
* 2013011187

## 原创性声明

此实验的代码 & 实验报告均为原创。


## 第一章 基础知识

### 1.1 了解图形工具箱

### 1.2 练习 Image file I/O 函数

在进行练习之前，我们先导入测试图像：

```matlab
load resource/hall
```

我们可以得到 `hall_color` 和 `hall_gray` 两个变量，其中测试图像是 `hall_color`。

#### 1.2a 画红圆

要画红圆，我们只需要先将测试图像复制一份，然后将圆圈内的点颜色全部置为 #FF0000 即可。代码如下： 

```matlab
%% add_circle: Add a red circle to the center of the given image
function circled_img = add_circle(img)
    circled_img = img;  % Copy the image.

    [y_max, x_max, ~] = size(circled_img);
    r = min(x_max, y_max) / 2;
    center = [(x_max + 1) / 2, (y_max + 1) / 2];

    for y = 1:y_max
        for x = 1:x_max
            if norm([x y] - center) <= r
                circled_img(y, x, :) = [255 0 0];
            end
        end
    end
```

调用函数并保存图像：

```matlab
imwrite(add_circle(hall_color), '../../report/hall_circal.bmp');
```

得到图像如下：

![Hall with a red circal](hall_circal.bmp)

#### 1.2b 黑白格涂色

为了将测试图像涂成国际象棋“黑白格”的样子，我们只需要对所有像素进行迭代，并计算出每块像素所属的格子。若为黑格子，我们只需将该位置改为黑色即可。具体代码如下：

```matlab
%% add_chess_board_mask: Add a mask like a chess board to a image
function masked_img = add_chess_board_mask(img)
    masked_img = img;  % Copy the image.

    [y_max, x_max, ~] = size(img);

    for y = 1:y_max
        for x = 1:x_max
            if mod(ceil(x / x_max * 8) + ceil(y / y_max * 8), 2) == 0
                masked_img(y, x, :) = [0 0 0];
            end
        end
    end
```

调用函数并保存图像：

```matlab
imwrite(add_chess_borad_mask(hall_color), '../../report/hall_chess_borad.bmp');
```

得到图像如下：

![Hall under a chess board](hall_chess_borad.bmp)

可以看到，以上两个图都达到了目标。

## 第二章 图像压缩编码

### 2.1 在变换域中实现预处理

由于变换域中的第一个分量便是直流分量，所以很明显这一步骤可以在变换域中进行。具体来说，由于 N = 8 时二维 DCT 变换的 DC 基底为 1/8，故只需要将 DC 分量减去 `128 / (1/8) = 1024` 即可。

需要注意的是，由于原矩阵元素类型为 `uint8`，故将其减去 128 前应将其转换成足够大的有符号数，例如 `int16`。

我们使用 `hall_gray` 的其中一块进行验证：

![Block](block.png)

```matlab
block = hall_gray(41:48, 65:72);
c = dct2(block);
c(1) = c(1) - 1024;  % Decimate DC component.

norm(c - dct2(int16(block) - 128))  % Compare two methods.
% ans =
%
%    2.1047e-13
```

可以看到，两种方法得到的变换域矩阵几乎完全相同。产生的一些误差可能来自于计算中的舍入误差。

### 2.2 实现二维 DCT

由公式 `C = D * P * DT` 可知，进行二维 DCT 的关键在于构造 DCT 算子 D。为此我们先定义函数 `trans_mat`：

```matlab
%% trans_mat: Construct NxN DCT transform matrix
function D = trans_mat(N)
    D = sqrt(2 / N) * cos([0:N-1]' * [1:2:2*N-1] * pi / (2 * N));
    D(1, :) = sqrt(1 / N);
```

然后我们便可以轻松进行二维 DCT 变换了：

```matlab
%% my_dct2: My implementation of dct2
function B = my_dct2(A)
    % DCT transform matrix.
    [row, col] = size(A);
    B = trans_mat(row) * double(A) * trans_mat(col)';
```

和内置函数 `dct2` 进行比较：

```matlab
norm(my_dct2(block) - dct2(block))
% ans =
%
%    7.9534e-13
```

可以看到误差极小，说明我们实现的二维 DCT 变换是正确的。

### 2.3 改变 DCT 系数

我们先来看一下 8x8 DCT 的基底：

![8x8 DCT](https://upload.wikimedia.org/wikipedia/commons/2/24/DCT-8x8.png)

可以发现，相对于前四列，右侧四列基底在横向上都有较高频变化。故若将右四列置为 0，恢复出的图像应在横向上没有高频分量。反之，若将左四列置为 0，则恢复出的图像在横向上应没有低频分量。

我们先对原来的 block 和 `hall_gray(17:24, 81:88)` 进行验证：

```matlab
c_right_zero = c;
c_left_zero = c;
c_right_zero(:, 5:8) = 0;
c_left_zero(:, 1:4) = 0;

subplot 311
imshow(block)
title Origin

subplot 312
imshow(uint8(idct2(c_right_zero) + 128))
title 'Zeros On The Right'

subplot 313
imshow(uint8(idct2(c_left_zero) + 128))
title 'Zeros On The Left'
```

![Zero columns](zero_cols.png)
![Zero columns (block2)](zero_cols_2.png)

可以看到，横向上的低频分量和高频分量被分离到了中下两幅图中，和我们的理论分析一致。

### 2.4 转置/旋转 DCT 系数

若对 DCT 系数转置，则横向与纵向分量系数互换，故恢复出的图像同样发生转置。

若将 DCT 系数旋转 90°，则大部分能量会转移到左下角，即纵向高频横向低频。

同样，若旋转 180°，则大部分能量会转移到右下角，即双向高频。

实际效果如下所示：

```matlab
subplot 221
imshow(uint8(idct2(c) + 128))
title Origin

subplot 222
imshow(uint8(idct2(c') + 128))
title Transpose

subplot 223
imshow(uint8(idct2(rot90(c)) + 128))
title 'Rotate 90 degree'

subplot 224
imshow(uint8(idct2(rot90(rot90(c))) + 128))
title 'Rotate 180 degree'
```

![Transpose / rotate coefficients](trans_rot.png)

和我们的理论分析一致。令人不解的是，左下图与右上图，左上图与右下图看起来竟然有些许相似。让我们看看换另一块的看看处理结果：

![Transpose / rotate coefficients (block 2)](trans_rot_2.png)

仍然和我们的理论分析一致，但相似性几乎消失了。之前的相似性可能是由频谱的巧妙分布，导致旋转后仍会形成条带，所以看起来和原图有些相似。

### 2.5

不考虑初值，差分编码系统的差分方程为：

    cD^(n) = -cD~(n) + cD~(n - 1)

即 `A = 1, B = [-1 1]`

使用 `freqz` 画出频率响应：

```matlab
freqz([-1 1], 1);
```

![Frequncy response](freqz.png)

可以看出，这是一个高通系统。这说明 DC 系数的低频频率分量更多，故经过高通系统后能量会有很大的衰减，从而达到信息压缩的目的。

### 2.6 DC 预测误差与 Category 的关系

从表中不难观察出，Category 的值即为预测误差所对应的 Magnitude 二进制表示的长度。具体关系为：

    Category = ceil(log2(|error| + 1))
### 2.8 分块，DCT & 量化

为了实现分块，DCT 和量化，我们编写了 `preprocess` 函数。其中具体完成的操作如下：

首先，为了之后的矩阵计算，我们先将传入的图像 `img` 转换为 `double`，同时将每个元素减去 128。

```matlab
img = double(img) - 128;  % Convert to double for matrix ops later.
```

然后，分块前我们确保图像的尺寸是 8 的倍数，若不是则用右下方元素填充：

```matlab
% Ensure row/col is a multiple of 8.
origin_size = size(img);
new_size = ceil(origin_size / 8) * 8;
left = new_size - origin_size;

img = [img,                            img(:, end) * ones(1, left(2))
       ones(left(1), 1) * img(end, :), img(end) * ones(left)];
```

然后我们便可以遍历所有块，对每个块进行 DCT ，量化和 Zig-Zag 遍历：

```matlab
out = zeros(64, numel(img) / 64);  % Placeholder for the output.

% Scanning blocks.
k = 1;
for row = 1:8:new_size(1)
    for col = 1:8:new_size(2)
        c = dct2(img(row:row+7, col:col+7));  % DCT.
        c = round(c ./ QTAB);                 % Quantize.
        out(:, k) = c(zigzag(8));             % Zig-Zag.
        k = k + 1;
    end
end
```

### 2.9 实现 JPEG 编码

我们刚刚实现的 `preprocess` 函数已经实现了分块，DCT 和量化的工作。所以为了完成 JPEG 编码工作，我们还需要计算 DC 系数流和 AC 系数流。

我们先来定义几个辅助函数：

`diff_encode`: 计算差分编码：

```matlab
%% diff_encode: Encode using differential coding.
function Y = diff_encode(X)
    Y = [0 X] - [X 0];  % X(n - 1) - X(n).
    Y = [X(1), Y(2:end-1)];
end
```

`amp2cate`: 根据幅度计算 Huffman 表中所属类别：

```matlab
%% amp2cate: Convert amp to category
function cate = amp2cate(amp)
    cate = ceil(log2(abs(amp) + 1));
```

`dec2_1s`: 将十进制转化为 1-补码

```matlab
%% dec2_1s: Convert decimal to 1's complement
function y = dec2_1s(dec)
    y = dec2bin(abs(dec)) - '0';
    if dec < 0
        y = 1 - y;  % Use 1's complement.
    end
```

然后我们便可以开始实现具体的编码函数了。

对于 DC 系数，我们先对其进行差分编码，然后依次对每个预测误差进行编码，形成码流。需要注意的是，对于 Category 为 0 的预测误差，其取值只有可能是 0，故不需要编码其 Magnitude。具体代码实现如下：

```matlab
%% encode_dc: Encode DC component
function DC_stream = encode_dc(DC, DCTAB)
    errors = diff_encode(DC);
    category = amp2cate(errors);

    DC_stream = [];

    for k = 1:length(errors)
        e = errors(k);
        row = category(k) + 1;

        huff = DCTAB(row, 2:1+DCTAB(row, 1));
        if e == 0
            DC_stream = [DC_stream huff];
        else
            DC_stream = [DC_stream huff dec2_1s(e)];
        end
    end

    DC_stream = DC_stream';
end
```

让我们用 例2.2 的样例测试一下：

```matlab
num2str(encode_dc([10, 8, 60], DCTAB)')
% ans =
%
% 1  0  1  1  0  1  0  0  1  1  1  0  1  1  1  0  0  0  1  0  1  1
```

结果与例题答案一致。

紧接着，我们对 AC 系数进行编码。我们依次对每一列进行处理，对每一列做如下操作：

1. 找到第一个非零元素，若没有且仍有数据说明该列末尾有 0，加入 EOB 标记后处理下一块；
2. 若 run 大于 15 则加入 ZRL，直到 run 小余等于 15；
3. 对 amp 进行编码，加入码流；
4. 去除这列中已处理的元素，返回步骤 1。

具体代码实现如下：

```matlab
%% encode_ac: Encode AC component
function AC_stream = encode_ac(AC, ACTAB)
    AC_stream = [];

    for k = 1:size(AC, 2)  % For every block.
        col = AC(:, k);

        amp_index = find(col, 1);  % Find first non-zero.
        while numel(amp_index)
            amp = col(amp_index);
            Run = amp_index - 1;

            % Reduce zeros.
            while Run > 15
                AC_stream = [AC_stream 1 1 1 1 1 1 1 1 0 0 1];
                Run = Run - 16;
            end

            % Encode run/size
            Size = amp2cate(amp);
            row = Run * 10 + Size;
            huff = ACTAB(row, 4:3+ACTAB(row, 3));

            AC_stream = [AC_stream huff dec2_1s(amp)];  % Add to stream.

            col(1:amp_index) = [];  % Delete this run/amp.
            amp_index = find(col, 1);  % Find next non-zero.
        end
        if length(col)  % Zero(s) not coded, insert EOB
            AC_stream = [AC_stream 1 0 1 0];
        end
    end

    AC_stream = AC_stream';
```

我们使用 例2.3 中数据进行检验：

```matlab
AC = [10 3 0 0 2 zeros(1, 20) 1 zeros(1, 37)]';
num2str(encode_ac(AC, ACTAB)')
% ans =
%
% 1  0  1  1  1  0  1  0  0  1  1  1  1  1  1  1  1  0  0  0  1  0  ...
% 1  1  1  1  1  1  1  1  0  0  1  1  1  1  0  1  1  1  1  0  1  0
```

结果也与样例一致。于是我们编写顶层函数调用预处理函数和编码函数：

```matlab
%% jpeg_encode: Encode an image using JPEG.
function [DC_stream, AC_stream, height, width] = jpeg_encode(img)
    load ../../resource/JpegCoeff

    [height, width] = size(img);  % Save the origin size.
    coefficients = preprocess(img, QTAB);

    DC_stream = encode_dc(coefficients(1, :), DCTAB);
    AC_stream = encode_ac(coefficients(2:end, :), ACTAB);
end
```

对 hall_gray 进行处理，并将结果保存在 `jpegcodes.mat` 中：

```matlab
[DC_stream, AC_stream, height, width] = jpeg_encode(hall_gray);
save jpegcodes DC_stream AC_stream height width
```

### 2.10 计算压缩比

由于输入的图像每个像素为 8 比特（`uint8`），输出的 AC, DC 码流每个元素为 1 比特，假设高度与宽度各使用 4 个字节存储，则压缩比可使用下公式计算：

```matlab
(prod(size(hall_gray)) * 8 + 64) / (length([DC_stream; AC_stream]) + 64)
% ans =
%
%     6.4109
```

故压缩比约为 6.41。

这里要注意的是，我们在分子和分母上都加上了图像大小信息。这是因为无论是编码前还是编码后，其对于显示图像都是必须的。

### 2.11 实现 JPEG 解码

为了实现 JPEG 的解码，我们首先需要从 DC 系数流和 AC 系数流中恢复出 `preprocess` 的输出，然后再进行 `preprocess` 的逆过程，便可以恢复出图像。

与前面类似，我们先定义几个辅助函数：

`huffman_decode`: 利用对应表，从码流首段解出第一个码字。

由于 Huffman 编码是前缀码，故为了判断是否匹配成功，只需要判断是否只有一行与码字完全匹配。故我们使用 `candidate` 数组保留可能匹配的码字，在增长码字长度的同时去掉不匹配的候选码字，最后只剩一个码字时返回其行号。具体实现如下：

```matlab
%% huffman_decode: Decode huffman code
function [index, len] = huffman_decode(codes, huffman_table)
    candidate = 1:size(huffman_table, 1);

    len = 0;
    while length(candidate) > 1
        len = len + 1;
        for k = 1:length(candidate)
            row = candidate(k);
            if codes(len) ~= huffman_table(row, len)
                candidate(k) = 0;  % Mark as unqualified.
            end
        end
        candidate(candidate == 0) = [];  % Eliminate unqualified.
    end

    index = candidate;
```

`decode_amp`: 对 Magnitude/Amplitude 进行解码。

```matlab
%% decode_amp: Decode mag/amp
function amp = decode_amp(code)
    if isempty(code)
        amp = 0;
    elseif code(1) == 0  % Nagetive.
        amp = -bin2dec(int2str(1 - code)');
    else
        amp = bin2dec(int2str(code)');
    end
```

然后我们开始实现核心解码函数。

首先对于 DC 码流，我们只需要循环解出每个系数，然后使用 `cumsum` 解差分编码即可。具体实现如下：

```matlab
%% decode_dc: Decode DC component
function DC = decode_dc(DC_stream, DCTAB, block_num)
    DC = zeros(1, block_num);
    huffman_table = DCTAB(:, 2:end);

    for block = 1:block_num
        [index, len] = huffman_decode(DC_stream, huffman_table);
        DC_stream(1:len) = [];  % Remove decoded.
        category = index - 1;   % category == code length.

        DC(block) = decode_amp(DC_stream(1:category));
        DC_stream(1:category) = [];  % Remove decoded.
    end

    DC = cumsum([DC(1), -DC(2:end)]);
```

对于 AC 码流，有几点需要注意的地方。首先，我们需要在 ACTAB 表中加入 ZRL 和 EOB 的项，保证 Huffman 表的完整性；其次，我们需要从 `huffman_decode` 函数返回的行号中解析出 `Run` 和 `Size`，并在 `Run = Size = 0`（即 EOB）的情况下做特殊处理。具体实现如下：

```matlab
%% decode_ac: Decode AC component
function AC = decode_ac(AC_stream, ACTAB, block_num)
    AC = zeros(63, block_num);
    huffman_table = [ACTAB(:, 4:end)
                     ones(1, 8) 0 0 1 zeros(1, 5)  % ZRL
                     1 0 1 0 zeros(1, 12)];  % EOB

    for block = 1:block_num
        k = 1;
        while k <= 63
            [index, len] = huffman_decode(AC_stream, huffman_table);
            AC_stream(1:len) = [];  % Remove decoded.
            [Run, Size] = decode_index(index);

            if Run == 0 & Size == 0  % EOB
                break  % Go to next block.
            end

            k = k + Run;  % Skip Run steps, because they are already 0s.

            AC(k, block) = decode_amp(AC_stream(1:Size));
            AC_stream(1:Size) = [];  % Remove decoded.
            k = k + 1;  % Skip amp.
        end
    end
end

%% decode_index: Decode index into Run & Size
function [Run, Size] = decode_index(index)
    if index <= 160
        Run = floor((index - 1) / 10);
        Size = mod(index - 1, 10) + 1;
    elseif index == 161  % ZRL
        Run = 15;
        Size = 0;
    else  % EOB
        Run = 0;
        Size = 0;
    end
end
```

这时我们已经解出了预处理后的 DC 与 AC 系数。紧接着，我们要实现 `preprocess` 的逆过程 `inv_preprocess`。我们依次对系数矩阵的每一列进行逆 Zig-Zag，反量化和 DCT 逆变换，最后给每个像素值加上 128 并转换为 `uint8` 类型。具体是实现如下：

```matlab
%% inv_preprocess: Inverse the preprocess
function [img] = inv_preprocess(pre_out, QTAB, height, width)
    img = zeros(ceil([height width] / 8) * 8);

    % Scanning blocks.
    k = 1;
    for row = 1:8:height
        for col = 1:8:width
            block = zeros(8, 8);

            block(zigzag(8)) = pre_out(:, k);          % Inverse Zig-Zag.
            block = block .* QTAB;                     % Inverse quantize.
            img(row:row+7, col:col+7) = idct2(block);  % Inverse DCT.

            k = k + 1;
        end
    end

    img = img(1:height, 1:width);  % Cut to the origin size.
    img = uint8(img + 128);
```

然后我们只需要用一个顶层函数调用这些函数，即可得到解码后的图像灰度值矩阵：

```matlab
%% jpeg_decode: decode a JPEG encoded image.
function img = jpeg_decode(DC_stream, AC_stream, height, width)
    load ../../resource/JpegCoeff

    block_num = prod(ceil([height width] / 8));

    DC = decode_dc(DC_stream, DCTAB, block_num);
    AC = decode_ac(AC_stream, ACTAB, block_num);

    img = inv_preprocess([DC; AC], QTAB, height, width);
end
```

为了测试编解码的效果，我们计算 PSNR，同时主观比较编码前后图像的差异：

```matlab
load jpegcodes
decoded_img = jpeg_decode(DC_stream, AC_stream, height, width);
psnr(decoded_img, hall_gray)
% ans =
%
%    31.1874
subplot 211
imshow(hall_gray);
title Origin
subplot 212
imshow(my_img);
title Decoded
```

![Compare origin & decoded](compare_decoded.png)

可以看到，PSNR 的值大约是 31.19，同时解码出的图像大致上与原图像相差无几。但仔细观察便可以发现，很多地方还是有编解码的痕迹。例如，大礼堂上方三角的边界变得不太清晰，而且有明显的分块感；大礼堂入口处柱子，以及两侧的树的细节都有失真。我们推测，这应该是由于这些地方的高频分量较大，故量化误差较大，所以失真较为严重。

不过总的来说，考虑到这张图像的大小只有 120x168，我们的编码再解码得到的图像确实与原图十分相似，达到了在视觉差异不太大的前提下有损压缩的目标。

### 2.12 减小量化步长后编解码

我们直接将传入编解码器中使用的参数由 `QTAB` 改为 `QTAB / 2`，用一样的方式编码和解码，得到结果如下

```matlab
(prod(size(hall_gray)) * 8 + 64) / (length([DC_stream; AC_stream]) + 64)
% ans =
%
%     4.4037
psnr(decoded_img, hall_gray)
% ans =
%
%    34.2067
```

![Halve QTAB](change_qtab.png)

可以看到，图像的压缩率有所下降（6.41 => 4.40），而 PSNR 则有所提升（31.19 => 34.21）。

而在图像上，我们也可以发现之前失真较大的一些细节变得清晰了不少，例如树叶的细节变得清晰，大礼堂上方三角的分块感也不那么明显了。这说明之前那些失真确实主要是量化误差导致的。

### 2.13 编解码电视机雪花图像

使用完全相同的流程，用标准量化步长进行编解码，得到结果如下：

```matlab
(prod(size(snow)) * 8 + 64) / (length([DC_stream; AC_stream]) + 64)
% ans =
%
%     3.5981
psnr(decoded_img, snow)
% ans =
%
%    22.9244
```

![Compare origin & decode snow](snow_compare_decoded.png)

可以看到，不仅图像的压缩率下降了不少（6.41 => 3.60），PSNR 也有很大幅度的减小（31.19 => 22.92）。同时，观察编码前后的图像可以发现，有些地方似乎有“抹平”的痕迹（如右方中间的一块）。

这也是高频分量量化误差的表现。因为，雪花图片的各交流分量的分布比较均匀，然而标准量化步长中高频的步长较大，故作用在雪花图像上时造成了较大的量化误差。这说明，标准量化误差不太适用于这种高频分量较高的图片。当然，值得一提的是，整体上来，原图和编解码后的图都十分混乱，所以也看不出太大差异…

## 第三章 信息隐藏

## 第四章 人脸识别
