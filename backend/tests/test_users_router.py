"""Users API surface."""

from app.main import create_app


def test_users_primary_route_registered():
    app = create_app()
    paths = [getattr(r, "path", "") for r in app.routes]
    assert "/users/primary" in paths
