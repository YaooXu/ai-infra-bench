"""Additional contract cases: cancel a waiting stream and resume it repeatedly."""
import gc,io,json,runpy,sys,weakref
sys.stdin=io.StringIO('{"seed": 94177}\n')
api=runpy.run_path('/tests/verify_retention.py',run_name='probe_helpers')
make_scheduler=api['make_scheduler'];make_request=api['make_request'];step=api['step']
RequestStatus=api['RequestStatus']

def run():
    gc.collect();gc.disable()
    observations=[]
    for cancellation in (True,False):
        scheduler=make_scheduler()
        req,payload=make_request('independent',tokens=list(range(51,61)),resumable=True)
        request_ref,payload_ref=weakref.ref(req),weakref.ref(payload)
        scheduler.add_request(req)
        expected=list(range(51,61))
        for turn in range(3):
            # Two sampled tokens; only the final EOS is dropped on continuation.
            step(scheduler,101+turn);step(scheduler,0)
            expected.append(101+turn)
            assert payload_ref() is not None
            if cancellation and turn==0:
                del req,payload
                scheduler.finish_requests('independent',RequestStatus.FINISHED_ABORTED)
                assert request_ref() is None and payload_ref() is None
                observations.append('waiting-stream-cancel-released')
                break
            update,data=make_request('independent',tokens=[201+turn,301+turn,401+turn],
                resumable=True,caching=False,media=False)
            expected += [201+turn,301+turn,401+turn]
            scheduler.add_request(update);del update,data
            assert list(req.all_token_ids)==expected
            manager=make_scheduler().kv_cache_manager
            assert manager.allocate_slots(req,num_new_tokens=req.num_tokens) is not None
            manager.cache_blocks(req,req.num_tokens);manager.free(req)
            fresh,_=make_request('fresh',tokens=expected)
            expected_hit=((len(expected)-1)//4)*4
            actual=manager.get_computed_blocks(fresh)[1]
            assert actual==expected_hit,(turn,expected_hit,actual)
            observations.append({'turn':turn,'cache_hit':actual})
        if not cancellation:
            # Cancel a resumed runnable stream as a separate ownership boundary.
            del req,payload
            scheduler.finish_requests('independent',RequestStatus.FINISHED_ABORTED)
            assert request_ref() is None and payload_ref() is None
            observations.append('resumed-stream-cancel-released')
    print(json.dumps(observations))
if __name__=='__main__':run()
