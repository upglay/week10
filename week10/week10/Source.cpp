#include <iostream>
#include <vector>

// linear regression < deep learning < machine learning
class LinearHypothesis
{
public:
	// linear hypothesis : y = a * x + b
	float a_ = 0.0f;
	float b_ = 0.0f;

	float getY(const float& x_input)
	{
		return a_ * x_input + b_; // returns y = a*x+b
	}
};

class Hypothesis
{
public:
	float a_ = 0.0f;
	float b_ = 0.0f;
	float c_ = 0.0f;

	float getY(const float& x_input)
	{
		return a_ * x_input * x_input + b_ * x_input + c_;
	}
};

const int num_data = 100000;

int main()
{
	LinearHypothesis lh;


	std::vector<float> tr_data;
	std::vector<float> re_data;
	// y = 3.0 * x + 10 + random_number
	for (int i = 0; i < num_data; i++)
	{
		const int random_value = rand() / 1000;
		tr_data.push_back(random_value);
		re_data.push_back(3.0 * random_value + 10 + random_value);
	}	

	for (int tr = 0; tr < 1; tr++)
	{
		for (int i = 0; i < num_data; i++)
		{
			// let's train our linear hypothesis to answer correctly!
			const float x_input = tr_data.at(i);
			const float y_output = lh.getY(x_input);
			const float y_target = re_data.at(i);
			const float error = y_output - y_target;
			// we can consider that our LH is trained well when error is 0 or small enough
			// we define squared error
			const float sqr_error = 0.5 * error * error; // always zero or positive

			// sqr_error = 0.5 * (a * x + b - y_target)^2
			// d sqr_error / da = 2*0.5*(a * x + b - y_target) * x; 
			// d sqr_error / db = 2*0.5*(a * x + b - y_target) * 1;
			const float dse_over_da = error * x_input;
			const float dse_over_db = error;

			const float lr = 0.000001; // small number
			lh.a_ -= dse_over_da * lr;
			lh.b_ -= dse_over_db * lr;
		}
		
	}

	// trained hypothesis
	float compare = 3;
	std::cout << "입력값 : "<< compare << " 원하는 값 : " << 3.0 * compare + 10 << " LinearHypothesis로 훈련된 결과 값 : " << lh.getY(compare) << std::endl << std::endl;

	//step2
	Hypothesis h;
	for (int i = 0; i < num_data; i++)
	{
		// let's train our linear hypothesis to answer correctly!
		const float x_input = tr_data.at(i);
		const float y_output = h.getY(x_input);
		const float y_target = re_data.at(i);
		const float error = y_output - y_target;
		// we can consider that our LH is trained well when error is 0 or small enough
		// we define squared error
		const float sqr_error = 0.5 * error * error; // always zero or positive
		
		// sqr_error = 0.5 * (a * x + b - y_target)^2
		// d sqr_error / da = 2*0.5*(a * x + b - y_target) * x * x; 
		// d sqr_error / db = 2*0.5*(a * x + b - y_target) * x;
		// d sqr_error / dc = 2*0.5*(a * x + b - y_target) * 1;
		const float dse_over_da = error * x_input * x_input;
		const float dse_over_db = error * x_input;
		const float dse_over_dc = error;

		const float lr = 0.000001; // small number
		h.a_ -= dse_over_da * lr;
		h.b_ -= dse_over_db * lr;
		h.c_ -= dse_over_dc;
	}

	std::cout << "입력값 : " << compare << " 원하는 값 : " << 3.0 * compare + 10 << " 2degree Hypothesis로 훈련된 값 : " << h.getY(compare) << std::endl;
	return 0;
}