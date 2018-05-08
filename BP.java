package BPNetWork;

import java.io.BufferedReader;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BP {

	private static final int IM = 4; // 输入层数量
	private static final int OM = 1; // 输出层数量
	private static final int RM = 8; // 隐含层数量
	private static final Path path = Paths.get("E:/Users/fanxin/Workspaces/MyEclipse 2017 CI/BP newwork/src/Iris.txt");
	private double learnRate = 0.5; // 学习速率
	private double thresholdHide[] = new double[RM];
	private double thresholdOut[] = new double[OM];
	private List<List<Double>> trainData = new ArrayList<List<Double>>();
	private List<List<Integer>> trainLabel = new ArrayList<List<Integer>>();
	private double[][] nom_data; // 归一化输入数据中的最大值和最小值
	private double Win[][] = new double[IM][RM]; // 输入到隐含连接权值
	private double Wout[][] = new double[RM][OM]; // 隐含到输出连接权值
	private double Ek[] = new double[OM];
	private double J = 0.1;
	double out[][] = new double[3][8];

	public static void main(String agrs[]) {
		BP bp = new BP();
		bp.train();
		bp.test();

	}

	public BP() {
		// 初始化权值和清零//
		readData(path);
		NormalizeData(trainData);
		for (int i = 0; i < trainData.size(); i++) {
			for (int j = 0; j < trainData.get(i).size(); j++) {
				trainData.get(i).set(j, Normalize(trainData.get(i).get(j), nom_data[j][0], nom_data[j][1]));
			}
		}
		// for(List<Double> list:trainData){
		// for(Double num:list){
		// System.out.print(num+" ");
		// }
		// System.out.println();
		// }
		InitNetWork();
	}

	// 初始化网络的权值和阈值
	public void InitNetWork() {
		// 初始化上一次权值量,范围为-0.5-0.5之间
		// in_hd_last = new double[IM][RM];
		// hd_out_last = new double[RM][OM];

		for (int i = 0; i < IM; i++)
			for (int j = 0; j < RM; j++) {
				int flag = 1; // 符号标志位(-1或者1)
				if ((new Random().nextInt(2)) == 1)
					flag = 1;
				else
					flag = -1;
				Win[i][j] = (new Random().nextDouble() / 2) * flag; // 初始化in-hidden的权值
			}

		for (int i = 0; i < RM; i++)
			for (int j = 0; j < OM; j++) {
				int flag = 1; // 符号标志位(-1或者1)
				if ((new Random().nextInt(2)) == 1)
					flag = 1;
				else
					flag = -1;
				Wout[i][j] = (new Random().nextDouble() / 2) * flag; // 初始化hidden-out的权值
			}

		// 阈值均初始化为0
		for (int k = 0; k < RM; k++)
			thresholdHide[k] = 0;

		for (int k = 0; k < OM; k++)
			thresholdOut[k] = 0;

	}

	public void train() {
		System.out.println("training");
		out = new double[3][8];
		for (int iter = 0; iter < 2000; iter++) {
			for (int cnd = 0; cnd < trainData.size(); cnd++) {
				// 第一层输入节点赋值

				for (int i = 0; i < IM; i++) {
					out[0][i] = trainData.get(cnd).get(i); // 为输入层节点赋值，其输入与输出相同
				}
				bpNetForwardProcess(trainLabel.get(cnd)); // 前向传播
				bpNetReturnProcess(); // 误差反向传播
				// System.out.println("This is the " + (iter + 1) + " th
				// trainning NetWork !");
				// System.out.println("All Samples Accuracy is " + J);
			}
		}
		System.out.println("training over" + " " + J);
	}

	public void test() {
		double count = 0;
		for (int cnd = 0; cnd < trainData.size(); cnd++) {
			// 第一层输入节点赋值

			for (int i = 0; i < IM; i++) {
				out[0][i] = trainData.get(cnd).get(i); // 为输入层节点赋值，其输入与输出相同
			}
			count += predict(trainLabel.get(cnd)); // 前向传播
		}
		System.out.println("预测准确率" + count / trainData.size());
	}

	public int predict(List<Integer> label) {
		bpNetForwardProcess(label); // 前向传播
		// 输出层S激活输出//
		boolean flag = true;
		for (int j = 0; j < OM; j++) {
			if ((out[2][j] > 0.5 && label.get(j) == 0) || (out[2][j] < 0.5 && label.get(j) == 1)) {
				flag = false;
				break;
			}
		}
		if (flag) {
			return 1;
		}
		return 0;
	}

	public void bpNetForwardProcess(List<Integer> label) {
		// 隐含层权值和计算//
		// 计算隐层节点的输出值
		for (int j = 0; j < RM; j++) {
			double v = -thresholdHide[j];
			for (int i = 0; i < IM; i++)
				v += Win[i][j] * out[0][i];
			out[1][j] = 1 / (1 + Math.exp(-v));
		}
		// 计算输出层节点的输出值
		for (int j = 0; j < OM; j++) {
			double v = -thresholdOut[j];
			for (int i = 0; i < RM; i++)
				v += Wout[i][j] * out[1][i];
			out[2][j] = 1 / (1 + Math.exp(-v));
		}
		// 计算输出与理想输出的偏差//
		for (int k = 0; k < OM; k++) {
			Ek[k] = out[2][k] - label.get(k);
		}
		J = 0;
		for (int k = 0; k < OM; k++) {
			J = J + Ek[k] * Ek[k] / 2.0;
		}

	}

	public void bpNetReturnProcess() {
		// 隐层到输出权值修正
		double g[] = new double[OM];
		for (int j = 0; j < OM; j++) {
			g[j] = Ek[j] * out[2][j] * (1 - out[2][j]);
		}
		for (int i = 0; i < RM; i++) {
			for (int j = 0; j < OM; j++) {
				Wout[i][j] += -learnRate * g[j] * out[1][i]; // 未加权值动量项
			}

		}

		double e[] = new double[RM];
		// 计算隐层的delta值
		for (int h = 0; h < RM; h++) {
			double t = 0;
			for (int j = 0; j < OM; j++)
				t += Wout[h][j] * g[j];
			e[h] = t * out[1][h] * (1 - out[1][h]);
		}

		// 输入层和隐含层之间权值和阀值调整
		for (int i = 0; i < IM; i++) {
			for (int h = 0; h < RM; h++) {
				Win[i][h] += -learnRate * e[h] * out[0][i]; // 未加权值动量项
			}
		}

		for (int j = 0; j < OM; j++)
			thresholdOut[j] += learnRate * g[j];
		for (int h = 0; h < RM; h++)
			thresholdHide[h] += learnRate * e[h];

	}

	/**
	 * 
	 * @author:凡鑫
	 * @data:2018年5月6日
	 * @time:下午6:48:15
	 * @param path
	 * @description:从txt中读数据
	 */
	public void readData(Path path) {
		try (BufferedReader reader = Files.newBufferedReader(path)) {
			String line = reader.readLine();
			int i = 0;
			while (line != null) {
				int j = 0;
				List<Double> tempList = new ArrayList<Double>();
				List<Integer> labelList = new ArrayList<Integer>();
				String data[] = line.split(",");
				for (; j < data.length - 1; j++) {
					tempList.add(Double.valueOf(data[j]));
				}
				if (data[j].equals("Iris-setosa"))
					labelList.add(0);
				else
					labelList.add(1);
				trainData.add(tempList);
				trainLabel.add(labelList);
				line = reader.readLine();
				i++;
			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}

	// 学习样本归一化,找到输入样本数据的最大值和最小值
	public void NormalizeData(List<List<Double>> trainData) {
		// 提前获得输入数据的个数
		int flag = 1;
		nom_data = new double[IM][2];
		for (List<Double> list : trainData) {
			for (int i = 0; i < list.size(); i++) {
				if (flag == 1) {
					nom_data[i][0] = Double.valueOf(list.get(i));
					nom_data[i][1] = Double.valueOf(list.get(i));
				} else {
					if (Double.valueOf(list.get(i)) > nom_data[i][0])
						nom_data[i][0] = Double.valueOf(list.get(i));
					if (Double.valueOf(list.get(i)) < nom_data[i][1])
						nom_data[i][1] = Double.valueOf(list.get(i));
				}
			}
			flag = 0;
		}
		/*
		 * for(int i=0;i<4;i++){ for(int j=0;j<2;j++){
		 * System.out.print(nom_data[i][j]+" "); } System.out.println(); }
		 */
	}

	// 归一化
	public double Normalize(double x, double max, double min) {
		double y = 0.1 + 0.8 * (x - min) / (max - min);
		return y;
	}

}
