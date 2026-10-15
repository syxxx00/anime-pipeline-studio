/* Read-only tracking. A timeout is not a failed generation and never resubmits it. */
(function(root){
  function create({readJob,readRuntime,onJob,onConnection=()=>{},onMissing=()=>{},
    onRuntimeUnavailable=()=>{},onRenderError=()=>{},setTimer=setTimeout,clearTimer=clearTimeout,interval=1500,timeout=15000}){
    let revision=0,timer=null,active=null,failures=0;
    const controllers=new Set();
    const terminal=new Set(['completed','failed','cancelled']);
    function render(job,runtime){try{onJob(job,runtime);}catch(error){onRenderError(error);}}
    async function bounded(read,ms){
      const controller=new AbortController();controllers.add(controller);
      const timeoutId=setTimer(()=>controller.abort(),ms);
      try{return await read(controller.signal);}
      finally{clearTimer(timeoutId);controllers.delete(controller);}
    }
    function stop(){revision++;active=null;clearTimer(timer);timer=null;for(const controller of controllers)controller.abort();controllers.clear();}
    async function tick(id,version){
      const current=()=>revision===version&&active===id;
      let delay=interval;
      try{
        const job=await bounded(signal=>readJob(id,signal),timeout);
        if(!current())return;
        failures=0;onConnection(null);render(job,null);
        if(terminal.has(job.status)){onRuntimeUnavailable(null);return;}
        if(readRuntime){
          try{
            const runtime=await bounded(signal=>readRuntime(signal),Math.min(timeout,5000));
            if(!current())return;onRuntimeUnavailable(null);render(job,runtime);
          }catch(error){if(current())onRuntimeUnavailable(error);}
        }
      }catch(error){
        if(!current())return;
        if(error.status===404){onMissing(id,error);return;}
        delay=Math.min(30000,2000*2**Math.min(failures++,4));
        onConnection({error,retryMs:delay,jobId:id});
      }
      if(current())timer=setTimer(()=>tick(id,version),delay);
    }
    function watch(id){stop();active=id;failures=0;onRuntimeUnavailable(null);return tick(id,revision);}
    return {watch,stop};
  }
  root.JobWatcher={create};if(typeof module!=='undefined')module.exports=root.JobWatcher;
})(globalThis);
