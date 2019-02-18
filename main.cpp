//=====================================================
// Evaluation of an integral using Monte Carlo Method
//=====================================================
#include <omp.h>
#include <iostream>
#include <math.h>
#include <random>
#include <ctime>
//=====================================================
using namespace std;
//=====================================================
struct Point
{
    Point(double x = 0.0, double y = 0.0)
    {
        this->x = x;
        this->y = y;
    }

    double x;
    double y;
};
//=====================================================
class RandomPointGenerator
{
public:
    RandomPointGenerator(double xMin, double xMax, double yMin, double yMax)
    {
        genX = mt19937 ();
        distX = uniform_real_distribution<> (xMin, xMax);

        genY = mt19937();
        distY = uniform_real_distribution<> (yMin, yMax);
    }

    RandomPointGenerator(int seedMultiplier, double xMin, double xMax, double yMin, double yMax)
    {
        genX = mt19937 (static_cast<unsigned int>(seedMultiplier * time(nullptr)));
        distX = uniform_real_distribution<> (xMin, xMax);

        genY = mt19937 (static_cast<unsigned int>(123 * seedMultiplier * time(nullptr)));
        distY = uniform_real_distribution<> (yMin, yMax);
    }

    void generate(Point& point)
    {
        point.x = distX(genX);
        point.y = distY(genY);
    }

private:

    mt19937 genY;
    uniform_real_distribution<> distY;
    mt19937 genX;
    uniform_real_distribution<> distX;
};
//=====================================================
#define SET_GENERATOR
//#define OUTPUT_POINTS


static double XMin = 0.0;
static double XMax = 0.0;
static double YMin = 0.0;
static double YMax = 0.0;
static double eps = 0.0;

static long minimalSampleNumber = 0;
static long acceptedSampleTotalNumber = 0;
static long sampleTotalNumber = 0;
static double integral = 0.0;

inline double f(double x);
bool defineYMinMax();
bool inputInitData();
double iterateIntegralCalculation();
void calculateIntegral();
void outputResult();

//=====================================================
int main()
{
    cout << "Evaluation of integral using Monte Carlo method." << endl;
    cout << "Function: x * cos(x) / sqrt(pow(x, 2.0) + 2.0)" << endl << endl;

    if(inputInitData() && defineYMinMax())
    {
        calculateIntegral();
        outputResult();
    }

    system("pause");
    return 0;
}
//=====================================================
double f(double x)
{
    return x * cos(x) / sqrt(pow(x, 2.0) + 2.0);
}
//=====================================================
bool defineYMinMax()
{
    int intervalCount = 1000;
    double dx = (XMax - XMin) / static_cast<double>(intervalCount);

    YMin = f(XMin);
    YMax = YMin;

#pragma omp parallel
    {
        double x;
        double fVal;
        double localMin = YMin;
        double localMax = YMax;

#pragma omp for
        for(int i = 0; i < intervalCount; i++)
        {
            x = XMin + static_cast<double>(i) * dx;
            fVal = f(x);
            if(localMax < fVal)
            {
                localMax = fVal;
            }
            else if(localMin > fVal)
            {
                localMin = fVal;
            }
        }

#pragma omp critical
        {
            if(YMax < localMax)
            {
                YMax = localMax;
            }

            if(YMin > localMin)
            {
                YMin = localMin;
            }
        }
    }

    if(YMin >= YMax)
    {
        cout << "Y axis margins are invalid!";
        return false;
    }

    // increase Y interval by 10 percent
    double dY = (YMax - YMin) / 20.0;
    YMax = YMax + dY;
    YMin = YMin - dY;

    cout << "Y Min: " << YMin << " Y Max: " << YMax << endl;

    return true;
}
//=====================================================
bool inputInitData()
{
    cout << "Input interval of integration" << endl;
    cout << "Min X: ";
    cin >> XMin;
    cout <<  "Max X: ";
    cin >> XMax;
    cout << "Minimal number of points: ";
    cin >> minimalSampleNumber;
    cout <<  "Acceptable error: ";
    cin >> eps;
    cout << endl;

    if(XMin >= XMax || eps <= 0.0 || minimalSampleNumber <= 0)
    {
        cout << "Initial data error!" << endl;
        return false;
    }

    return true;
}
//=====================================================
void calculateIntegral()
{
    int threadCount = omp_get_max_threads();
    long sampleNumber = minimalSampleNumber;

    double previousIntegral = 0.0;
    bool previousIntegralIsValid = false;
    bool work = true;
    int iterationCount = 0;

#pragma omp parallel num_threads(threadCount) shared(sampleNumber)
    {
#ifdef SET_GENERATOR
        RandomPointGenerator generator(omp_get_thread_num() + 1,
                                       XMin, XMax, YMin, YMax);
#else
        RandomPointGenerator generator(XMin, XMax, YMin, YMax);
#endif

        Point point;
        double fVal = 0.0;
        int acceptedSampleCount = 0;

        do
        {
#pragma omp for
            for(long i = 0; i < sampleNumber; i++)
            {
                generator.generate(point);
                fVal = f(point.x);

#ifdef OUTPUT_POINTS
                //  output of point coordinates
#pragma omp critical
                {
                    cout << "Thread: " << omp_get_thread_num() << " i " << i
                         << " Point x " << point.x << " y " << point.y << endl;
                }
#endif
                if(fVal > 0)
                {
                    if(point.y > 0 && point.y < fVal)
                    {
                        // point in positive part
                        acceptedSampleCount++;
                    }
                }
                else if(fVal < 0)
                {
                    if(point.y < 0 && point.y > fVal)
                    {
                        // point in negative part
                        acceptedSampleCount--;
                    }
                }
                // if fVal == 0  continue
            }

#pragma omp critical
            {
                // add together accepted point counts;
                acceptedSampleTotalNumber += acceptedSampleCount;
                acceptedSampleCount = 0;
            }

#pragma omp barrier


#pragma omp single
            {
                sampleTotalNumber += sampleNumber;
                integral = iterateIntegralCalculation();

                cout << "Iteration " << ++iterationCount << endl;
                cout << "Total number of points: " << sampleTotalNumber << endl;
                cout << "Integral evaluation: " << integral << endl << endl;

                if(previousIntegralIsValid && abs(previousIntegral - integral) < eps)
                {
                    // integral calculated
                    work = false;
                }

                if(work)
                {
                    // check pointTotalNumber overflow
                    if(LONG_MAX - sampleNumber > sampleTotalNumber)
                    {
                        previousIntegral = integral;
                        previousIntegralIsValid = true;
                        sampleNumber = sampleTotalNumber;
                    }
                    else
                    {
                        // pointTotalNumber overflow
                        cout << "Error: total number of samples overflow!" << endl;
                        work = false;
                    }
                }
            }
        }while(work);
    }
}
//=====================================================
void outputResult()
{
    cout << endl;
    cout << "Final evaluation of the integral: " <<  integral << endl;
    cout << endl;
}
//=====================================================
double iterateIntegralCalculation()
{
    double integralPersentage = static_cast<double>(acceptedSampleTotalNumber)
            / static_cast<double>(sampleTotalNumber);
    double integral = (XMax - XMin) * (YMax -YMin) * integralPersentage;

    return integral;
}
//=====================================================



