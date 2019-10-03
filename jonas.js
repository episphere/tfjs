console.log('jonas.js loaded')

jonas=function(){
    console.log('jonas fun initialized at '+Date())
}

jonas.getIris=async function(){ // await jonas.getIris() will place data at jonas.irisData and also return it
    jonas.irisData=await (await fetch('https://episphere.github.io/ai/data/iris.json')).json()
    jonas.irisData=jonas.dt2tab(jonas.irisData)
    console.log('iris data retrieved and also stored at jonas.irisData')
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
    const attrs = Object.keys(dt[0])
    const tb={parmsIn:attrs.slice(0,-1)}

    tb.x=dt.map(x=>{
        return tb.parmsIn.map(lb=>x[lb])        
    })
    tb.parmOut=Object.keys(dt[0]).slice(-1)[0]
    tb.y=dt.map(x=>x[tb.parmOut]) 
    tb.labels=[...new Set(tb.y)] // unique labels
    tb.y=tb.y.map(y=>tb.labels.map(yi=>(yi==y)))
    return tb
}
