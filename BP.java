package BPNetWork;

import java.io.BufferedReader;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BP {

	private static final int IM = 4; // ���������
	private static final int OM = 1; // ���������
	private static final int RM = 8; // ����������
	private static final Path path = Paths.get("E:/Users/fanxin/Workspaces/MyEclipse 2017 CI/BP newwork/src/Iris.txt");
	private double learnRate = 0.5; // ѧϰ����
	private double thresholdHide[] = new double[RM];
	private double thresholdOut[] = new double[OM];
	private List<List<Double>> trainData = new ArrayList<List<Double>>();
	private List<List<Integer>> trainLabel = new ArrayList<List<Integer>>();
	private double[][] nom_data; // ��һ�����������е����ֵ����Сֵ
	private double Win[][] = new double[IM][RM]; // ���뵽��������Ȩֵ
	private double Wout[][] = new double[RM][OM]; // �������������Ȩֵ
	private double Ek[] = new double[OM];
	private double J = 0.1;
	double out[][] = new double[3][8];

	public static void main(String agrs[]) {
		BP bp = new BP();
		bp.train();
		bp.test();

	}

	public BP() {
		// ��ʼ��Ȩֵ������//
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

	// ��ʼ�������Ȩֵ����ֵ
	public void InitNetWork() {
		// ��ʼ����һ��Ȩֵ��,��ΧΪ-0.5-0.5֮��
		// in_hd_last = new double[IM][RM];
		// hd_out_last = new double[RM][OM];

		for (int i = 0; i < IM; i++)
			for (int j = 0; j < RM; j++) {
				int flag = 1; // ���ű�־λ(-1����1)
				if ((new Random().nextInt(2)) == 1)
					flag = 1;
				else
					flag = -1;
				Win[i][j] = (new Random().nextDouble() / 2) * flag; // ��ʼ��in-hidden��Ȩֵ
			}

		for (int i = 0; i < RM; i++)
			for (int j = 0; j < OM; j++) {
				int flag = 1; // ���ű�־λ(-1����1)
				if ((new Random().nextInt(2)) == 1)
					flag = 1;
				else
					flag = -1;
				Wout[i][j] = (new Random().nextDouble() / 2) * flag; // ��ʼ��hidden-out��Ȩֵ
			}

		// ��ֵ����ʼ��Ϊ0
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
				// ��һ������ڵ㸳ֵ

				for (int i = 0; i < IM; i++) {
					out[0][i] = trainData.get(cnd).get(i); // Ϊ�����ڵ㸳ֵ���������������ͬ
				}
				bpNetForwardProcess(trainLabel.get(cnd)); // ǰ�򴫲�
				bpNetReturnProcess(); // ���򴫲�
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
			// ��һ������ڵ㸳ֵ

			for (int i = 0; i < IM; i++) {
				out[0][i] = trainData.get(cnd).get(i); // Ϊ�����ڵ㸳ֵ���������������ͬ
			}
			count += predict(trainLabel.get(cnd)); // ǰ�򴫲�
		}
		System.out.println("Ԥ��׼ȷ��" + count / trainData.size());
	}

	public int predict(List<Integer> label) {
		bpNetForwardProcess(label); // ǰ�򴫲�
		// �����S�������//
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
		// ������Ȩֵ�ͼ���//
		// ��������ڵ�����ֵ
		for (int j = 0; j < RM; j++) {
			double v = -thresholdHide[j];
			for (int i = 0; i < IM; i++)
				v += Win[i][j] * out[0][i];
			out[1][j] = 1 / (1 + Math.exp(-v));
		}
		// ���������ڵ�����ֵ
		for (int j = 0; j < OM; j++) {
			double v = -thresholdOut[j];
			for (int i = 0; i < RM; i++)
				v += Wout[i][j] * out[1][i];
			out[2][j] = 1 / (1 + Math.exp(-v));
		}
		// ������������������ƫ��//
		for (int k = 0; k < OM; k++) {
			Ek[k] = out[2][k] - label.get(k);
		}
		J = 0;
		for (int k = 0; k < OM; k++) {
			J = J + Ek[k] * Ek[k] / 2.0;
		}

	}

	public void bpNetReturnProcess() {
		// ���㵽���Ȩֵ����
		double g[] = new double[OM];
		for (int j = 0; j < OM; j++) {
			g[j] = Ek[j] * out[2][j] * (1 - out[2][j]);
		}
		for (int i = 0; i < RM; i++) {
			for (int j = 0; j < OM; j++) {
				Wout[i][j] += -learnRate * g[j] * out[1][i]; // δ��Ȩֵ������
			}

		}

		double e[] = new double[RM];
		// ���������deltaֵ
		for (int h = 0; h < RM; h++) {
			double t = 0;
			for (int j = 0; j < OM; j++)
				t += Wout[h][j] * g[j];
			e[h] = t * out[1][h] * (1 - out[1][h]);
		}

		// ������������֮��Ȩֵ�ͷ�ֵ����
		for (int i = 0; i < IM; i++) {
			for (int h = 0; h < RM; h++) {
				Win[i][h] += -learnRate * e[h] * out[0][i]; // δ��Ȩֵ������
			}
		}

		for (int j = 0; j < OM; j++)
			thresholdOut[j] += learnRate * g[j];
		for (int h = 0; h < RM; h++)
			thresholdHide[h] += learnRate * e[h];

	}

	/**
	 * 
	 * @author:����
	 * @data:2018��5��6��
	 * @time:����6:48:15
	 * @param path
	 * @description:��txt�ж�����
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

	// ѧϰ������һ��,�ҵ������������ݵ����ֵ����Сֵ
	public void NormalizeData(List<List<Double>> trainData) {
		// ��ǰ����������ݵĸ���
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

	// ��һ��
	public double Normalize(double x, double max, double min) {
		double y = 0.1 + 0.8 * (x - min) / (max - min);
		return y;
	}

}
