/**
 * Build `AxialFlowNet_方法说明.docx` with native Word equations.
 *
 * This machine has no Python, no pandoc and no LibreOffice, so the .docx is
 * assembled directly: a zip containing [Content_Types].xml, _rels/.rels,
 * docProps/*, word/styles.xml and word/document.xml.  Formulas are emitted as
 * OMML (`m:oMath`) so they appear as editable Word equations rather than
 * flattened text.
 *
 * Run:  bun scripts/build_methods_docx.js --out <file.docx>
 */
import { writeFileSync, statSync } from "node:fs";

const FONT_LATIN = "Cambria Math";
const FONT_CJK = "宋体";
const FONT_BODY = "Times New Roman";
const FONT_BODY_CJK = "宋体";

// --------------------------------------------------------------------- helpers
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function rpr(bold = false, italic = false, latin = FONT_LATIN) {
  const parts = [`<w:rFonts w:ascii="${latin}" w:hAnsi="${latin}" w:eastAsia="${FONT_CJK}"/>`];
  if (bold) parts.push("<w:b/>");
  if (italic) parts.push("<w:i/>");
  return `<w:rPr>${parts.join("")}</w:rPr>`;
}

/** math text run */
const t = (txt, italic = false, bold = false) =>
  `<m:r>${rpr(bold, italic)}<m:t xml:space="preserve">${esc(txt)}</m:t></m:r>`;

/** italic identifier */
const mi = (name) => t(name, true);

/** upright number */
const mn = (v) => t(v, false);

const sub = (base, s) => `<m:sSub><m:sSubPr><m:ctrlPr>${rpr()}</m:ctrlPr></m:sSubPr><m:e>${base}</m:e><m:sub>${s}</m:sub></m:sSub>`;
const sup = (base, s) => `<m:sSup><m:sSupPr><m:ctrlPr>${rpr()}</m:ctrlPr></m:sSupPr><m:e>${base}</m:e><m:sup>${s}</m:sup></m:sSup>`;
const frac = (num, den) =>
  `<m:f><m:fPr><m:ctrlPr>${rpr()}</m:ctrlPr></m:fPr><m:num>${num}</m:num><m:den>${den}</m:den></m:f>`;
const func = (name, arg) =>
  `<m:func><m:funcPr><m:ctrlPr>${rpr()}</m:ctrlPr></m:funcPr><m:fName>${t(name)}</m:fName><m:e>${arg}</m:e></m:func>`;
const delim = (inner, beg = "[", end = "]") =>
  `<m:d><m:dPr><m:begChr m:val="${beg}"/><m:endChr m:val="${end}"/><m:ctrlPr>${rpr()}</m:ctrlPr></m:dPr><m:e>${inner}</m:e></m:d>`;
/** square root (degree hidden) */
const rad = (radicand) =>
  `<m:rad><m:radPr><m:degHide m:val="1"/><m:ctrlPr>${rpr()}</m:ctrlPr></m:radPr>` +
  `<m:deg/><m:e>${radicand}</m:e></m:rad>`;
const omath = (body) => `<m:oMath>${body}</m:oMath>`;

/** parenthesised call: mean(...), ReLU(...), tanh(...) */
const call = (name, inner) => func(name, delim(inner, "(", ")"));
/** mean|expr| */
const meanAbs = (inner) => call("mean", delim(inner, "|", "|"));
/** L1 norm: ‖expr‖₁ */
const nrm1 = (inner) =>
  `<m:sSub><m:sSubPr><m:ctrlPr>${rpr()}</m:ctrlPr></m:sSubPr>` +
  `<m:e>${delim(inner, "\u2016", "\u2016")}</m:e><m:sub>${mn(1)}</m:sub></m:sSub>`;

/** centred display equation paragraph */
function displayMath(body, numbered = null) {
  const tag = numbered ? `(${numbered})` : "";
  const runs = tag ? `<m:r>${rpr()}<m:t xml:space="preserve">&#8195;&#8195;${esc(tag)}</m:t></m:r>` : "";
  return (
    `<w:p><w:pPr><w:spacing w:before="160" w:after="160"/><w:jc w:val="center"/>` +
    `<w:rPr>${rpr()}</w:rPr></w:pPr><m:oMathPara><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>` +
    `<m:oMath>${body}${runs}</m:oMath></m:oMathPara></w:p>`
  );
}

// ------------------------------------------------------------------ paragraphs
function para(text, style = "", bold = false, size = 24) {
  const ppr = style ? `<w:pStyle w:val="${style}"/>` : "";
  const r = `${rpr(false, false, FONT_BODY).replace(FONT_LATIN, FONT_BODY)}<w:sz w:val="${size}"/>` +
    (bold ? "<w:b/>" : "");
  return `<w:p><w:pPr>${ppr}</w:pPr><w:r><w:rPr>${r}</w:rPr><w:t xml:space="preserve">${esc(text)}</w:t></w:r></w:p>`;
}

const heading = (level, text) =>
  `<w:p><w:pPr><w:pStyle w:val="Heading${level}"/></w:pPr><w:r><w:t>${esc(text)}</w:t></w:r></w:p>`;

function bullet(text) {
  const r = rpr(false, false, FONT_BODY).replace(FONT_LATIN, FONT_BODY);
  return `<w:p><w:pPr><w:ind w:left="420" w:hanging="240"/></w:pPr><w:r><w:rPr>${r}</w:rPr>` +
    `<w:t xml:space="preserve">&#8226;&#8195;${esc(text)}</w:t></w:r></w:p>`;
}

function table(headers, rows) {
  const widths = headers.length === 2 ? [3600, 5400] : [1800, 2400, 2400, 2400].slice(0, headers.length);
  const out = ['<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr><w:tblGrid>'];
  for (const w of widths) out.push(`<w:gridCol w:w="${w}"/>`);
  out.push("</w:tblGrid>");
  out.push('<w:tr><w:trPr><w:tblHeader/></w:trPr>');
  headers.forEach((h, i) => {
    out.push(`<w:tc><w:tcPr><w:tcW w:w="${widths[i]}" w:type="dxa"/></w:tcPr>` +
      `<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/>${rpr(false, false, FONT_BODY).replace(FONT_LATIN, FONT_BODY)}</w:rPr>` +
      `<w:t>${esc(h)}</w:t></w:r></w:p></w:tc>`);
  });
  out.push('</w:tr>');
  for (const row of rows) {
    out.push("<w:tr>");
    row.forEach((c, i) => {
      out.push(`<w:tc><w:tcPr><w:tcW w:w="${widths[i]}" w:type="dxa"/></w:tcPr>` +
        `<w:p><w:pPr>${i === 0 ? "" : '<w:jc w:val="center"/>'}</w:pPr>` +
        `<w:r><w:rPr>${rpr(false, false, FONT_BODY).replace(FONT_LATIN, FONT_BODY)}</w:rPr>` +
        `<w:t xml:space="preserve">${esc(c)}</w:t></w:r></w:p></w:tc>`);
    });
    out.push("</w:tr>");
  }
  out.push("</w:tbl>");
  out.push("<w:p/>"); // a paragraph after a table is required by Word
  return out.join("");
}


// ---------------------------------------------------------------------- content
function buildBody() {
  const B = [];
  const add = (x) => B.push(x);

  // ---- shared math symbols
  const IhatB = sub(mi("\u00ce"), t("B"));    // Î_B
  const IhatA = sub(mi("\u00ce"), t("A"));    // Î_A
  const ItildeA = sub(mi("\u0128"), t("A"));  // Ĩ_A
  const ItildeB = sub(mi("\u0128"), t("B"));  // Ĩ_B
  const IA = sub(mi("I"), t("A"));
  const IB = sub(mi("I"), t("B"));
  const GA = sub(mi("G"), t("A"));
  const GB = sub(mi("G"), t("B"));
  const TEX = mi("\u03a4");
  const LAM = (s) => sub(mi("\u03bb"), t(s));
  const LL = (s) => sub(mi("\u2112"), t(s));  // ℒ
  const gcall = (G, arg) => G + t("(") + arg + t(")");

  // ---- title
  add(`<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="240" w:after="120"/></w:pPr>` +
      `<w:r><w:rPr>${rpr(true, false, FONT_BODY).replace(FONT_LATIN, FONT_BODY)}<w:sz w:val="36"/></w:rPr>` +
      `<w:t>AxialFlowNet:几何流场引导的双向图像转换网络</w:t></w:r></w:p>`);
  add(`<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="360"/></w:pPr>` +
      `<w:r><w:rPr>${rpr(false, false, FONT_BODY).replace(FONT_LATIN, FONT_BODY)}<w:sz w:val="28"/></w:rPr>` +
      `<w:t>网络结构与损失函数说明</w:t></w:r></w:p>`);

  add(para("本文描述 AxialFlowNet 的网络结构、空间变换方式与训练目标。文中所有结构、损失定义与参数取值均按 " +
           "models/axialflownet_model.py、models/flow_head.py 与 models/axial_d_networks.py 的实现给出," +
           "参数取值读自各训练 run 的 train_opt.txt 与 inference_config.json。"));
  add(para("方法面向连续切片成像中随深度累积的轴向拉伸问题:该问题的修正本质上是空间形变,而不是全局的强度重映射。" +
           "因此网络将生成过程组织为几何变换与外观修正两个相互协作的子过程,并在结构上把二者分开,使位移场可以被单独约束与检视。"));

  // ================================================================= 1
  add(heading(1, "一、网络总体架构"));
  add(para("设两个非配对图像域为 A 域(轴向拉伸图像)与 B 域(去拉伸参考图像),输入记为 I_A 与 I_B。" +
           "两个生成器 G_A、G_B 分别学习 A→B 与 B→A 的映射:"));
  add(displayMath(
    IhatB + t(" = ") + gcall(GA, IA) + t(" ,\u2003\u2003") +
    IhatA + t(" = ") + gcall(GB, IB), 1));
  add(para("生成图像再经反向生成器重建回原域:"));
  add(displayMath(
    ItildeA + t(" = ") + gcall(GB, IhatB) + t(" ,\u2003\u2003") +
    ItildeB + t(" = ") + gcall(GA, IhatA), 2));
  add(para("循环一致性约束要求重建结果尽量接近原始输入,即 Ĩ_A ≈ I_A、Ĩ_B ≈ I_B。" +
           "每个生成器由共享残差主干与两个并行输出分支组成:几何流场分支预测稠密位移场,经可微分空间采样对输入图像做非刚性变换;" +
           "纹理残差分支预测有界残差,补偿变换后剩余的强度与纹理差异。完整网络包含:"));
  for (const x of [
    "双向图像生成器 G_A 与 G_B(共享残差主干 + 双分支输出);",
    "几何流场预测分支与纹理残差预测分支;",
    "基于流场的空间变换与图像融合模块;",
    "轴向双分支判别器,每个方向一个实例;",
    "对抗、结构、轴向单调性与流场正则化联合损失。"]) add(bullet(x));

  // ================================================================= 2
  add(heading(1, "二、生成器:共享主干与双分支"));
  add(para("两个生成器结构相同、参数独立。主干输出共享特征 F,再由流场分支与纹理分支分别预测位移场与像素残差," +
           "两个分支因此基于同一组特征表示学习几何变化与外观修正。"));

  add(heading(2, "2.1 共享特征主干"));
  add(para("主干按如下顺序处理输入图像:反射填充(3)→ 7×7 卷积 → InstanceNorm → ReLU → 两级跨步卷积下采样 → " +
           "9 个残差块 → 两级转置卷积上采样。主干输出 64 通道、全分辨率的共享特征 F,同时供两个分支使用。"));
  add(table(["阶段", "输出通道", "空间分辨率(相对输入)"], [
    ["初始 7×7 卷积", "64", "1/1"],
    ["第 1 次下采样", "128", "1/2"],
    ["第 2 次下采样", "256", "1/4"],
    ["9× 残差块", "256", "1/4"],
    ["第 1 次上采样", "128", "1/2"],
    ["第 2 次上采样", "64", "1/1"],
    ["共享特征 F", "64", "1/1"]]));
  add(para("主干只负责提供多层次特征表示,不直接决定最终输出。反射填充用于补充边界信息并减轻零填充造成的边缘效应;" +
           "残差块在低分辨率特征空间中提取图像特征;两级上采样把特征恢复到全分辨率,使两个分支都能逐像素给出输出。"));

  add(heading(2, "2.2 几何流场分支"));
  add(para("流场分支以共享特征 F 为输入,输出两通道位移场:"));
  add(displayMath(
    mi("u") + delim(mi("x") + t(", ") + mi("y"), "(", ")") + t(" = ") +
    delim(sub(mi("u"), mi("x")) + t(" ,\u2003") + sub(mi("u"), mi("y")), "(", ")"), 3));
  add(para("其中 u_x 与 u_y 分别表示沿图像宽度与高度方向的位移分量。FlowHead 由三层 3×3 卷积构成" +
           "(64→32→32→2),层间使用 LeakyReLU(0.2) 激活,末层输出经 tanh 限制在 [-1, 1] 内。" +
           "最后一层卷积的权重与偏置均以零初始化,因此训练初期位移严格为零,生成过程退化为恒等变换;" +
           "位移场随训练逐步增长,网络先学习外观修正,再学习空间形变。"));

  add(heading(2, "2.3 基于流场的空间变换"));
  add(para("得到位移场后,网络调用 warp_with_flow 对输入图像做空间变换。变换采用反向采样:输出图像每个像素依据归一化采样网格," +
           "从输入图像的对应位置取值。网格由恒等网格与位移场叠加得到,采样使用双线性插值,边界以边缘像素填充:"));
  add(displayMath(
    sub(mi("\u00ce"), t("warp")) + t(" = ") + func("W", delim(mi("I") + t(" , ") + mi("u"), "(", ")")) +
    t(" = ") + call("grid_sample", mi("I") + t(" , ") + mi("g")), 4));
  add(para("位移以像素为单位换算为归一化坐标偏移:"));
  add(displayMath(
    sub(mi("u"), mi("x")) + t(" \u21a6 ") + frac(t("2") + sub(mi("u"), mi("x")), mi("W")) +
    t(" ,\u2003\u2003") +
    sub(mi("u"), mi("y")) + t(" \u21a6 ") + frac(t("2") + sub(mi("u"), mi("y")), mi("H")), 5));
  add(para("由于归一化坐标的一个单位对应 W(或 H)个像素,位移数值 v 恰好对应 v 个像素的空间平移。" +
           "因此 tanh 的 [-1, 1] 界对应 1 像素的位移上限,而流场幅度阈值 τ 是以像素为单位的软上限:" +
           "小于 τ 的局部位移不受惩罚,只有更大的局部形变才被抑制。该换算关系经数值验证确定:以均匀 u_x = 1.0 " +
           "对图像中的一段列块做变换,其质心位移实测为 1.00 像素。"));

  add(heading(2, "2.4 纹理残差分支"));
  add(para("纹理分支同样接收共享特征 F,经反射填充(3)、7×7 卷积(64→输出通道数)与 tanh 得到有界残差:"));
  add(displayMath(
    TEX + t(" = ") + mi("b") + t(" \u00b7 ") +
    call("tanh", sub(mi("Conv"), t("7\u00d77")) + delim(mi("F"), "(", ")")), 6));
  add(para("其中 b 为缩放常数。tanh 将输出限制在 [-1, 1] 内,再乘以 b 后残差位于 [-b, b]。该上限是该分支保持补充性的关键:" +
           "它可以修正强度与局部纹理,但不能独立重写整幅图像。b 已固化在权重中,因此重新载入 checkpoint 时必须使用训练时的取值。"));

  add(heading(2, "2.5 融合与最终输出"));
  add(para("最终图像由空间变换结果与纹理残差相加,再截断到合法像素范围:"));
  add(displayMath(
    sub(mi("\u00ce"), t("B")) + t(" = ") +
    call("Clamp", func("W", delim(mi("I") + t(" , ") + mi("u"), "(", ")")) + t(" + ") + TEX + t(" , -1 , 1")), 7));
  add(para("网络输出并非解码器直接给出的整幅图像,而是变换后的输入图像与有界残差之和。" +
           "在流场为零的初始状态下,生成结果近似为输入图像加残差,几何变换与外观修正因此在训练中先后建立。"));

  // ================================================================= 3
  add(heading(1, "三、判别器"));
  add(heading(2, "3.1 双分支结构"));
  add(para("模型为每个转换方向实例化一个 AxialMultiscaleDiscriminator。该判别器由两个并行分支组成,全部卷积层均施加谱归一化," +
           "两个分支的输出沿通道维拼接后由两层 1×1 卷积融合为单通道预测图:"));
  add(table(["分支", "卷积核与步长", "作用"], [
    ["空间分支", "4×4 卷积,步长 2/2/2/1,通道 64→128→256→512", "70×70 感受野,捕获局部二维纹理"],
    ["轴向分支", "3×8 卷积,步长 (1,2)/(1,2)/(1,2)/(1,1),通道同上", "沿图像宽度方向拉长的卷积核,捕获长程共现结构"],
    ["融合层", "1×1 卷积(1024→256)→ LeakyReLU → 1×1 卷积(256→1)", "合并两分支,输出单通道预测图"]]));
  add(para("两个分支在同一输入分辨率上并行工作。轴向分支的拉长卷积核沿拉伸作用的方向跨越多个像素," +
           "使判别器能够约束各向同性小块统计所平均掉的残余方向性;若两个分支的输出尺寸不同,轴向分支的输出会先双线性插值到空间分支的尺寸再融合。"));
  add(para("两层下采样后,分支沿长轴方向分别捕获中等尺度与更长的结构,通道数逐层翻倍(64→128→256→512)。" +
           "判别器输出为单通道预测图,空间尺寸与 70×70 PatchGAN 一致。"));

  add(heading(2, "3.2 判别器与图像域的对应关系"));
  add(para("两个判别器实例的命名依据其判别的数据,而非映射方向:"));
  add(table(["实例", "判别对象"], [
    ["netD_A", "真实 B 域图像 与 生成图像 Î_B"],
    ["netD_B", "真实 A 域图像 与 生成图像 Î_A"]]));
  add(para("即 netD_A 是 B 域的判别器,对应 G_A 的输出。网络结构图应按这一实际数据流连接。"));

  // ================================================================= 4
  add(heading(1, "四、损失函数"));
  add(para("生成器的训练目标由对抗损失、结构保持损失、轴向单调性损失、流场正则化项与身份映射损失组成。"));

  add(heading(2, "4.1 对抗损失"));
  add(para("对抗损失由 GANLoss 计算,两个方向求和:"));
  add(displayMath(
    LL("adv") + t(" = ") + sup(LL("GAN"), t("A\u2192B")) + t(" + ") + sup(LL("GAN"), t("B\u2192A")), 8));
  add(para("具体形式由 gan_mode 决定,两个 released run 均使用 vanilla(最小二乘形式)。"));

  add(heading(2, "4.2 结构保持损失"));
  add(para("该损失对循环重建结果计算,由像素 L1、SSIM 项与 Sobel 边缘 L1 三部分相加:"));
  add(displayMath(
    LL("struct") + t(" = ") + nrm1(ItildeA + t(" \u2212 ") + IA) + t(" + ") +
    sub(mi("\u03b1"), t("SSIM")) + delim(t("1 \u2212 ") + call("SSIM", ItildeA + t(" , ") + IA), "(", ")") + t(" + ") +
    sub(mi("\u03b1"), t("edge")) + nrm1(mi("\u2207") + ItildeA + t(" \u2212 ") + mi("\u2207") + IA), 9));
  add(para("默认 α_SSIM = 1.0、α_edge = 0.5。仅有像素误差时网络容易收敛到模糊的重建结果;" +
           "SSIM 项约束局部结构与对比度,Sobel 项直接比较梯度边缘图,使重建结果保留切片图像中承担结构信息的边缘。" +
           "该损失在两个方向上分别计算,并以对应方向的权重 λ_A、λ_B 加权。"));

  add(heading(2, "4.3 轴向单调性损失"));
  add(para("该损失作用于重建误差 E = Ĩ − I,惩罚其在图像高度方向上出现的负向差分:"));
  add(displayMath(
    LL("mono") + t(" = ") + call("mean", call("ReLU", t("\u2212") + delim(sub(mi("E"), t("y+1")) + t(" \u2212 ") + sub(mi("E"), t("y")), "(", ")"))), 10));
  add(para("即只有相邻两行之间误差下降的情形被惩罚,误差随行号非下降的分布不受约束。默认权重 2.0。" +
           "该约束以图像纵轴与切片堆叠方向一致为前提,用于抑制重建误差沿轴向的振荡。"));

  add(heading(2, "4.4 流场正则化"));
  add(para("平滑项为总变分形式,分别惩罚水平与垂直方向的相邻位移差:"));
  add(displayMath(
    LL("flow_smooth") + t(" = ") + frac(mn(1), mn(2)) +
    delim(meanAbs(sub(mi("\u0394"), mi("x")) + mi("u")) + t(" + ") + meanAbs(sub(mi("\u0394"), mi("y")) + mi("u")), "(", ")"), 11));
  add(para("该正则鼓励邻近位置的位移变化平缓,抑制局部折叠与散斑,同时允许平滑的整体位移。"));
  add(para("单调性项仅对垂直位移分量的负差分施加惩罚:"));
  add(displayMath(
    LL("flow_mono") + t(" = ") +
    call("mean", call("ReLU", t("\u2212") + sub(mi("\u0394"), mi("y")) + sub(mi("u"), mi("y")))), 12));
  add(para("该正则表达的是垂直位移分量沿纵轴单调不减的先验,与切片堆叠方向上的轴向校正趋势相对应。"));
  add(para("幅度项对超过阈值 τ 的位移幅度施加软惩罚。位移幅度定义为"));
  add(displayMath(
    mi("m") + delim(mi("x") + t(", ") + mi("y"), "(", ")") + t(" = ") +
    rad(sup(sub(mi("u"), mi("x")), mn(2)) + t(" + ") + sup(sub(mi("u"), mi("y")), mn(2)) + t(" + ") + mi("\u03b5")), 13));
  add(displayMath(
    LL("flow_mag") + t(" = ") + call("mean", call("ReLU", mi("m") + t(" \u2212 ") + mi("\u03c4"))), 14));
  add(para("幅度不超过 τ 时不产生惩罚,超过后随超出幅度线性增加,属于软约束。" +
           "由 §2.3 的换算可知 τ 以像素为单位,因此该项直接限制局部形变的像素幅度。"));

  add(heading(2, "4.5 身份映射损失"));
  add(para("当 lambda_identity > 0 时,将目标域图像送入对应生成器,并要求输出接近原图:"));
  add(displayMath(
    LL("id") + t(" = ") + LAM("B") + LAM("id") + nrm1(gcall(GA, IB) + t(" \u2212 ") + IB) + t(" + ") +
    LAM("A") + LAM("id") + nrm1(gcall(GB, IA) + t(" \u2212 ") + IA), 15));
  add(para("该约束减少输入已属目标域时的不必要外观改变。默认 lambda_identity = 0.5,仅在训练配置启用身份损失时参与计算。"));

  add(heading(2, "4.6 总目标"));
  add(para("生成器总目标为各项的加权和,其中流场正则项先对两个方向取平均,结构损失与循环重建项按方向加权:"));
  add(displayMath(
    LL("G") + t(" = ") + LL("adv") + t(" + ") +
    LAM("struct") + delim(LAM("A") + sub(mi("\u2112"), t("struct,A")) + t(" + ") + LAM("B") + sub(mi("\u2112"), t("struct,B")), "(", ")") + t(" + ") +
    LAM("mono") + delim(sub(mi("\u2112"), t("mono,A")) + t(" + ") + sub(mi("\u2112"), t("mono,B")), "(", ")"), 16));
  add(displayMath(
    t("+ ") + LAM("fs") + frac(mn(1), mn(2)) + delim(sub(mi("\u2112"), t("fs,AB")) + t(" + ") + sub(mi("\u2112"), t("fs,BA")), "(", ")") +
    t(" + ") + LAM("fm") + frac(mn(1), mn(2)) + delim(sub(mi("\u2112"), t("fm,AB")) + t(" + ") + sub(mi("\u2112"), t("fm,BA")), "(", ")"), 17));
  add(displayMath(
    t("+ ") + LAM("fmag") + frac(mn(1), mn(2)) + delim(sub(mi("\u2112"), t("fmag,AB")) + t(" + ") + sub(mi("\u2112"), t("fmag,BA")), "(", ")") +
    t(" + ") + LL("id"), 18));
  add(para("结构项同时携带方向权重 λ_A/λ_B 与 λ_struct,单调性项与流场正则项只携带各自权重;" +
           "循环重建的像素 L1 由结构损失中的第一项承担。"));
  add(para("两个 released run 的权重取值如下,读自各 run 的 train_opt.txt;" +
           "texture_bound 与 flow_mag_threshold 未记录在该文件中,取自各 run 的 inference_config.json:"));
  add(table(["权重", "3axial_AxialFlowNet_100ep", "xianliti_AxialFlowNet_200ep"], [
    ["λ_A / λ_B", "10.0 / 10.0", "10.0 / 10.0"],
    ["λ_identity", "0.5", "0.5"],
    ["λ_struct", "1.0", "1.0"],
    ["α_SSIM / α_edge", "1.0 / 0.5", "1.0 / 0.5"],
    ["λ_mono", "2.0", "2.0"],
    ["λ_flow_smooth", "2.0", "2.0"],
    ["λ_flow_mono", "5.0", "5.0"],
    ["λ_flow_mag", "1.0", "1.0"],
    ["texture_bound (b)", "0.3", "0.15"],
    ["flow_mag_threshold (τ)", "0.3", "0.4"],
    ["gan_mode", "vanilla", "vanilla"],
    ["ngf / ndf / norm", "64 / 64 / instance", "64 / 64 / instance"]]));

  // ================================================================= 5
  add(heading(1, "五、训练流程"));
  add(para("训练由生成器更新与判别器更新交替完成。生成器阶段:前向计算两个方向的生成结果、流场与循环重建," +
           "据此计算对抗、结构、单调性与流场正则化损失;随后冻结判别器参数,反向传播总目标并更新两个生成器。"));
  add(para("判别器阶段:重新启用判别器参数,从历史图像池采样生成图像,与真实域图像一起输入对应判别器," +
           "对真实样本与生成样本分别计算判别损失并取平均,再更新两个判别器。"));
  add(para("优化器为 Adam,β₁ = 0.5,两个 run 的 batch_size 均为 1。3axial_AxialFlowNet_100ep 以 resize_and_crop" +
           "(286 → 256)预处理,恒定学习率 80 epoch 后线性衰减 20 epoch,合计 100 epoch,单 run 采样上限 200 张;" +
           "xianliti_AxialFlowNet_200ep 以 preprocess = none 保持原生分辨率、经反射填充对齐到 4 的倍数," +
           "恒定学习率 200 epoch 后线性衰减 50 epoch,合计 250 epoch。"));

  return B.join("");
}


// ---------------------------------------------------------------------- package
const CONTENT_TYPES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>`;

const RELS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>`;

const DOC_RELS = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>`;

const CORE = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:title>AxialFlowNet:几何流场引导的双向图像转换网络 — 网络结构与损失函数说明</dc:title>
<dc:creator>Wu, An</dc:creator>
<cp:lastModifiedBy>Wu, An</cp:lastModifiedBy>
</cp:coreProperties>`;

const APP = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>bun</Application></Properties>`;

const STYLES = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:styleId="Heading1" w:name="heading 1"><w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:styleId="Heading2" w:name="heading 2"><w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="200" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:styleId="Heading3" w:name="heading 3"><w:pPr><w:outlineLvl w:val="2"/><w:spacing w:before="160" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:styleId="TableGrid" w:name="Table Grid"><w:tblPr><w:tblBorders>
<w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/>
<w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/>
</w:tblBorders></w:tblPr></w:style>
</w:styles>`;

const buildDocument = () =>
  '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' +
  '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' +
  'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">' +
  `<w:body>${buildBody()}</w:body></w:document>`;

// ------------------------------------------------------------------- zip writer
const CRC_TABLE = (() => {
  const tbl = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    tbl[n] = c;
  }
  return tbl;
})();

function crc32(buf) {
  let c = 0 ^ -1;
  for (let i = 0; i < buf.length; i++) c = (c >>> 8) ^ CRC_TABLE[(c ^ buf[i]) & 0xff];
  return (c ^ -1) >>> 0;
}

/** Write a .docx. Entries are stored uncompressed; docx readers accept that. */
function writeZip(path, entries) {
  const chunks = [];
  const central = [];
  let offset = 0;
  const enc = (s) => new TextEncoder().encode(s);

  for (const [name, data] of entries) {
    const nameBytes = enc(name);
    const dataBytes = typeof data === "string" ? enc(data) : data;
    const crc = crc32(dataBytes);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);        // version needed
    local.setUint16(6, 0, true);         // flags
    local.setUint16(8, 0, true);        // method: store
    local.setUint16(10, 0, true);        // mod time
    local.setUint16(12, 0x21, true);     // mod date (1980-01-01)
    local.setUint32(14, crc, true);
    local.setUint32(18, dataBytes.length, true);
    local.setUint32(22, dataBytes.length, true);
    local.setUint16(26, nameBytes.length, true);
    local.setUint16(28, 0, true);
    const localBuf = Buffer.concat([Buffer.from(local.buffer), Buffer.from(nameBytes), Buffer.from(dataBytes)]);
    chunks.push(localBuf);

    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(8, 0, true);
    cd.setUint16(10, 0, true);
    cd.setUint16(12, 0, true);
    cd.setUint16(14, 0x21, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, dataBytes.length, true);
    cd.setUint32(24, dataBytes.length, true);
    cd.setUint16(28, nameBytes.length, true);
    cd.setUint16(30, 0, true);
    cd.setUint16(32, 0, true);
    cd.setUint16(34, 0, true);
    cd.setUint32(42, offset, true); // relative offset of local header (byte 42, not 38)
    central.push(Buffer.concat([Buffer.from(cd.buffer), Buffer.from(nameBytes)]));
    offset += localBuf.length;
  }

  const cdBuf = Buffer.concat(central);
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(4, 0, true);
  eocd.setUint16(6, 0, true);
  eocd.setUint16(8, entries.length, true);
  eocd.setUint16(10, entries.length, true);
  eocd.setUint32(12, cdBuf.length, true);
  eocd.setUint32(16, offset, true);
  eocd.setUint16(20, 0, true);
  writeFileSync(path, Buffer.concat([...chunks, cdBuf, Buffer.from(eocd.buffer)]));
  return statSync(path).size;
}

// ------------------------------------------------------------------ validation
/** Structural well-formedness check: tags nest and balance, entities valid. */
function checkXml(xml, label) {
  const stack = [];
  const re = /<(\/?)([A-Za-z_][\w:.-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>|<!--[\s\S]*?-->|[^<]+/g;
  let m;
  while ((m = re.exec(xml)) !== null) {
    const [whole, closing, name, , selfClose] = m;
    if (whole.startsWith("<!--") || !closing && !name) continue;
    if (!closing) {
      if (!selfClose) stack.push(name);
    } else {
      const top = stack.pop();
      if (top !== name) throw new Error(`${label}: mismatched </${name}> (open was <${top ?? "none"}>)`);
    }
  }
  if (stack.length) throw new Error(`${label}: unclosed <${stack.join(">, <")}>`);
  // stray ampersands
  const bad = xml.match(/&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)/);
  if (bad) throw new Error(`${label}: bare & at ${bad.index}`);
  return true;
}

function main() {
  const out = Bun.env.DOCX_OUT ?? "AxialFlowNet_方法说明.docx";
  const doc = buildDocument();

  for (const [label, xml] of [["document.xml", doc], ["[Content_Types].xml", CONTENT_TYPES],
                              ["styles.xml", STYLES]]) checkXml(xml, label);

  const size = writeZip(out, [
    ["[Content_Types].xml", CONTENT_TYPES],
    ["_rels/.rels", RELS],
    ["docProps/core.xml", CORE],
    ["docProps/app.xml", APP],
    ["word/styles.xml", STYLES],
    ["word/_rels/document.xml.rels", DOC_RELS],
    ["word/document.xml", doc],
  ]);

  for (const tag of ["<m:oMath>", "<m:oMathPara>", "<m:sSub>", "<m:f>", "<m:func>",
                     "<m:sSup>", "<m:rad>", "<m:dPr>", "</w:document>", "<w:tbl>"]) {
    if (!doc.includes(tag)) throw new Error(`missing ${tag}`);
  }
  console.log(`OK ${out}  size=${size}B  equations=${(doc.match(/<\/m:oMath>/g) || []).length} ` +
              `paragraphs=${(doc.match(/<w:p>/g) || []).length} tables=${(doc.match(/<w:tbl>/g) || []).length}`);
}

main();
