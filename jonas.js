console.log('jonas.js loaded')

jonas=function(){
    console.log('jonas fun initialized at '+Date())
}

jonas.getIris=async function(){ // await jonas.getIris() will place data at jonas.irisData and also return it
    jonas.irisData=await (await fetch('https://episphere.github.io/ai/data/iris.json')).json()
    return jonas.irisData
}

jonas.getTrainTest=async function(dt,p){ // returns [xtrain, ytrain, ytrain, ytest] with a 0-1 sampling fraction
    dt = dt || jonas.irisData
    if(!dt){
        dt = await jonas.getIris()
    }
    debugger
    return '[xtrain, ytrain, ytrain, ytest]'
}

jonas.dt2tab=function(dt){ // creates a tabular data frame
    dt=dt || jonas.irisData
    const tb={labels:Object.keys(dt[0])}
    tb.data=dt.map(x=>{
        return tb.labels.map(lb=>x[lb])        
    })
    
    return tb
}
