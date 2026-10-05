const $ = id => document.getElementById(id);

const trace = [
  {title:'Build the record',file:'a2/scripts/a2common.py · content()',question:'Which two corpus fields become model-visible text? Which metadata fields would leak a label?',code:'return raw_text + "\\nTable:\\n" + table_json\n# source_kind and source_url are excluded'},
  {title:'Choose a mode',file:'llmbox/startllm.py · _DISPATCH',question:'Where does a one-turn request branch from a multi-turn chat session?',code:'mode=batch → Modes.run_batch → BatchRunner.run\nSUBMODES = generate | structured | tools | guarded | fewshot'},
  {title:'Generate and measure',file:'llmbox/src/generation.py · _generate_measured()',question:'Which setting changes the response length? What changes when sampling is enabled?',code:'apply_chat_template(messages)\nmodel.generate(max_new_tokens, do_sample, temperature, top_p)\nreturn text + tokens + latency + finish_reason'},
  {title:'Check the response',file:'llmbox/src/structured.py · structured_generate()',question:'If JSON matches the schema, what have you learned? What still requires a reference label?',code:'first_json_object(text) → Pydantic validation\ninvalid → validation feedback → retry\nvalidity and task correctness are separate measurements'},
  {title:'Guard and route',file:'llmbox/src/guardrail.py · Guardrail.run()',question:'What is checked before generation, and what is checked after? Which failures should be reviewed?',code:'check_input → allow/block\ninner mode → check_output → allow/review/block\nlog hashes and flags, not raw text'},
  {title:'Compare and decide',file:'a2/scripts/score.py · score(); a2/scripts/cost.py · model()',question:'Which measurements depend on human labels, and which can be read directly from run logs?',code:'responses + independent labels → validity, exact match, fields\nlatency + tokens + measured human time + sourced costs → cost model'}
];

function selectTrace(i) {
  const t = trace[i];
  $('trace-file').textContent = t.file;
  $('trace-title').textContent = t.title;
  $('trace-question').textContent = t.question;
  $('trace-code').textContent = t.code;
  document.querySelectorAll('#trace-tabs button').forEach((b,j) => b.setAttribute('aria-selected', String(i === j)));
}
trace.forEach((t,i) => { const b=document.createElement('button'); b.type='button'; b.setAttribute('role','tab'); b.textContent=`${String(i+1).padStart(2,'0')} · ${t.title}`; b.onclick=()=>selectTrace(i); $('trace-tabs').append(b); });
selectTrace(0);

async function loadState() {
  try {
    const res=await fetch('/api/state'); const s=await res.json();
    $('status').textContent = `${s.labels_ready ? 'Reference labels imported' : 'Reference labels pending'} · ${s.probes_ready ? 'Part D probes present' : 'Part D probes pending'} · ${s.a1_lab_available ? 'A1 lab available' : 'A1 lab unavailable'}`;
    $('run-grid').replaceChildren();
    for (const [name,run] of Object.entries(s.runs)) {
      const card=document.createElement('article'); card.className='run-card';
      const title=document.createElement('h3'); title.textContent=name.replaceAll('_',' ');
      const count=document.createElement('div'); count.className='count'; count.textContent=String(run.rows);
      const detail=document.createElement('p'); detail.textContent=run.metrics ? `valid ${Math.round(run.metrics.valid_output_rate*100)}% · exact ${Math.round(run.metrics.exact_match*100)}%` : 'Responses saved · scoring pending';
      card.append(title,count,detail); $('run-grid').append(card);
    }
    if(s.partd){
      const pending=!(s.partd.probe_design?.human_adjudication_complete);
      const heading=document.createElement('h3'); heading.textContent=`Part D · ${pending?'automatic rules':'adjudication included'}`;
      const note=document.createElement('p'); note.className='hint'; note.textContent=pending?'Complete blind adjudication before drawing conclusions. Catch rate and over-refusal must be read together.':'Catch rate and over-refusal must be read together.';
      const table=document.createElement('table');
      const header=document.createElement('tr');
      for(const label of ['Category','n','Baseline successes','Guarded successes','Catch / over-refusal']){const cell=document.createElement('th');cell.textContent=label;header.append(cell);} table.append(header);
      for(const [category,entry] of Object.entries(s.partd.by_category)){
        const row=document.createElement('tr');
        for(const value of [category,entry.n,entry.baseline_success??'—',entry.guarded_success??'—',entry.catch_rate===undefined?`${Math.round(entry.over_refusal_rate*100)}% over-refusal`:`${Math.round(entry.catch_rate*100)}% catch`]){const cell=document.createElement('td');cell.textContent=String(value);row.append(cell);} table.append(row);
      }
      $('safety-results').replaceChildren(heading,note,table);
    }
    const candidate=s.runs.C3_guarded;
    const latency=candidate.metrics?.latency_s?.mean;
    if(latency!==undefined&&!$('model-seconds').value) $('model-seconds').value=latency;
  } catch { $('status').textContent='Could not read local run state.'; }
}
loadState();

let prior=null;
$('run-form').addEventListener('submit',async event=>{
  event.preventDefault();
  const button=$('run-button'); button.disabled=true; button.textContent='Running…';
  $('run-state').textContent='Loading or generating locally…';
  const request={record:$('record').value,model:$('model').value,mode:$('mode').value,max_new_tokens:Number($('tokens').value),do_sample:$('sampling').value==='true',temperature:Number($('temperature').value),top_p:Number($('top-p').value)};
  try {
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    const result=await response.json(); if (!response.ok) throw new Error(result.error||`HTTP ${response.status}`);
    $('output').textContent=result.text||'(empty response)';
    $('run-state').textContent=result.blocked?'Blocked':result.review?'Review flagged':'Completed';
    const trace=result.tool_calls?.length?{tool_calls:result.tool_calls}:result.guard?{guard:result.guard}:null;
    $('trace-details').hidden=!trace;
    $('run-trace').textContent=trace?JSON.stringify(trace,null,2):'';
    $('measurements').replaceChildren();
    for(const [label,value] of [['Prompt tokens',result.prompt_tokens],['Completion tokens',result.completion_tokens],['Latency',`${result.latency_s}s`],['Retries',result.retries],['Tool calls',result.tool_calls?.length],['Finish',result.finish_reason]]){
      if(value===undefined) continue; const chip=document.createElement('span'); chip.textContent=`${label}: ${value}`; $('measurements').append(chip);
    }
    const comparison=prior ? `Previous run: ${prior.mode} / ${prior.model}, ${prior.prompt_tokens}+${prior.completion_tokens} tokens, ${prior.latency_s}s. Current run: ${request.mode} / ${request.model}, ${result.prompt_tokens}+${result.completion_tokens} tokens, ${result.latency_s}s. What caused the difference?` : 'Change one setting or mode, then run the same record again. Compare outputs and measurements.';
    $('compare').textContent=comparison;
    prior={...request,...result};
  } catch(error){$('run-state').textContent='Run failed';$('output').textContent=error.message;}
  finally{button.disabled=false;button.textContent='Run local model';}
});

$('cost-form').addEventListener('submit',event=>{
  event.preventDefault();
  const n=id=>Number($(id).value);
  const manual=n('manual-seconds'),review=n('review-seconds'),usable=n('usable-percent')/100,
        model=n('model-seconds'),staff=n('staff-cost'),compute=n('compute-cost'),
        setup=n('setup-hours'),volume=n('annual-records');
  const manualDollars=manual*staff/3600;
  const apiDollars=(review+(1-usable)*manual)*staff/3600+model*compute/3600;
  const saving=manualDollars-apiDollars;
  const breakEven=saving>0?Math.ceil(setup*staff/saving):null;
  $('cost-output').textContent=`Manual: $${manualDollars.toFixed(4)} per record\nAPI + review: $${apiDollars.toFixed(4)} per record\nSaving: $${saving.toFixed(4)} per record\nBreak-even: ${breakEven===null?'none at these assumptions':`${breakEven} records`}\nAnnual net after setup: $${(saving*volume-setup*staff).toFixed(2)}`;
});

const reviewEvents=[];
$('review-submit').addEventListener('click',()=>{
  const action=$('review-action').value;
  if(!action){$('review-log').textContent='Choose an action first.';return;}
  reviewEvents.push({fixture:'hypothetical_department_not_in_source',actor:'practice_reviewer',action,applied_update:false});
  $('review-log').textContent=reviewEvents.map((event,i)=>`${i+1}. ${JSON.stringify(event)}`).join('\n');
});
