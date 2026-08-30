use axum::Router;
use tokio_util::sync::CancellationToken;
use vllm_server::{Config, serve_with_router_extension};

#[allow(dead_code)]
async fn typecheck_public_extension_api(config: Config, shutdown: CancellationToken) {
    serve_with_router_extension(config, shutdown, |router: Router| router)
        .await
        .expect("extended router server should preserve the normal Result contract");
}

#[test]
fn extension_callback_has_router_to_router_shape() {
    fn extension(router: Router) -> Router {
        router
    }

    let _ = extension as fn(Router) -> Router;
}
