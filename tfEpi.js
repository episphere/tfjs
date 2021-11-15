console.log('tfEpi.js loaded');

tfEpi={}

tfEpi.Date=Date()

tfEpi.getCars=async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );
  const carsData = await carsDataResponse.json();
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower
    }))
    .filter((car) => car.mpg != null && car.horsepower != null);

  return cleaned;
}

tfEpi.run = async function (cars) {
  // Load and plot the original input data that we are going to train on.
  cars=cars||await tfEpi.getCars()
  const values = cars.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    {name: 'Horsepower v MPG'},
    {values},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
  );

  // More code will be added below
}


if(typeof(define)!='undefined'){
    define(tfEpi)
}