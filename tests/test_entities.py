"""entities 层单元测试：SoA 对象池（ArmorPool / BoxPool / ProjectilePool）。

覆盖：spawn/destroy 生命周期、free_stack 复用、路径系统、扩容、批量销毁。
"""

from __future__ import annotations

import numpy as np
import pytest

from Airis.lulilo.entities.pools import (
    MAX_ARMOR,
    MAX_BOXES,
    MAX_FORMATIONS,
    MAX_PROJ,
    ArmorPool,
    BoxPool,
    FormationAIPool,
    PartType,
    ProjectilePool,
    ProjectileType,
)


# ---------------------------------------------------------------------------
# ArmorPool
# ---------------------------------------------------------------------------


class TestArmorPool:
    """ArmorPool 测试套件。"""

    @pytest.fixture
    def pool(self) -> ArmorPool:
        return ArmorPool()

    def test_spawn_basic(self, pool: ArmorPool) -> None:
        """spawn 应返回有效索引并设置位置。"""
        idx = pool.spawn(100.0, 200.0, 50.0)
        assert idx >= 0
        assert pool.active[idx]
        assert pool.pos[idx, 0] == pytest.approx(100.0)
        assert pool.pos[idx, 1] == pytest.approx(200.0)
        assert pool.pos[idx, 2] == pytest.approx(50.0)

    def test_spawn_defaults(self, pool: ArmorPool) -> None:
        """spawn 默认参数应正确。"""
        idx = pool.spawn(0, 0, 0)
        assert pool.half[idx, 0] == pytest.approx(1.0)
        assert pool.half[idx, 1] == pytest.approx(3.0)
        assert pool.hull_idx[idx] == -1  # 未绑定 HULL box（part-health-design.md §2.2）
        assert pool.yaw[idx] == pytest.approx(0.0)

    def test_destroy_releases_index(self, pool: ArmorPool) -> None:
        """destroy 应标记 inactive 并归还索引。"""
        idx = pool.spawn(0, 0, 0)
        pool.destroy(idx)
        assert not pool.active[idx]
        # 索引应被复用
        idx2 = pool.spawn(1, 1, 1)
        assert idx2 == idx, "销毁后的索引应被立即复用"

    def test_active_count(self, pool: ArmorPool) -> None:
        """active_count 应正确计数。"""
        assert pool.active_count == 0
        idx1 = pool.spawn(0, 0, 0)
        assert pool.active_count == 1
        idx2 = pool.spawn(1, 1, 1)
        assert pool.active_count == 2
        pool.destroy(idx1)
        assert pool.active_count == 1
        pool.destroy(idx2)
        assert pool.active_count == 0

    def test_active_indices(self, pool: ArmorPool) -> None:
        """active_indices 应返回所有存活索引。"""
        a = pool.spawn(0, 0, 0)
        b = pool.spawn(1, 1, 1)
        c = pool.spawn(2, 2, 2)
        pool.destroy(b)
        indices = pool.active_indices()
        assert set(indices.tolist()) == {a, c}

    def test_set_waypoints(self, pool: ArmorPool) -> None:
        """set_waypoints 应正确存储路标点。"""
        idx = pool.spawn(0, 0, 0)
        wps = [(10, 20), (30, 40), (50, 60)]
        pool.set_waypoints(idx, wps)
        assert pool.wp_count[idx] == 3
        assert pool.wp_idx[idx] == 0
        assert pool.wp[idx, 0, 0] == pytest.approx(10)
        assert pool.wp[idx, 0, 1] == pytest.approx(20)
        assert pool.wp[idx, 2, 0] == pytest.approx(50)

    def test_set_waypoints_max_clip(self, pool: ArmorPool) -> None:
        """超过 8 个路标点时应裁剪到 8 个。"""
        idx = pool.spawn(0, 0, 0)
        wps = [(float(i), float(i)) for i in range(20)]
        pool.set_waypoints(idx, wps)
        assert pool.wp_count[idx] == 8
        assert pool.wp[idx, 7, 0] == pytest.approx(7)

    def test_set_waypoints_clears_old(self, pool: ArmorPool) -> None:
        """重新设置路标点应清除旧路径。"""
        idx = pool.spawn(0, 0, 0)
        pool.set_waypoints(idx, [(1, 1), (2, 2)])
        pool.set_waypoints(idx, [(3, 3)])
        assert pool.wp_count[idx] == 1
        assert pool.wp[idx, 0, 0] == pytest.approx(3)

    def test_spawn_fills_capacity(self, pool: ArmorPool) -> None:
        """装满预分配容量后应能继续 spawn（触发扩容）。"""
        # 先销毁全部再 spawn MAX_ARMOR 个
        indices = [pool.spawn(float(i), 0, 0) for i in range(MAX_ARMOR)]
        assert len(indices) == MAX_ARMOR
        assert pool.active_count == MAX_ARMOR
        # 扩容边界：再 spawn 一个
        extra = pool.spawn(999, 999, 999)
        assert pool.active_count == MAX_ARMOR + 1
        assert pool.active[extra]

    # ---- Phase 6c AI 字段（docs/phase6-ai-design.md §1.1）----

    def test_spawn_ai_fields_defaults(self, pool: ArmorPool) -> None:
        """spawn 后 AI 字段默认值应正确（未编队/无目标/通信满）。"""
        idx = pool.spawn(0, 0, 0)
        assert pool.side[idx] == 0
        assert pool.role[idx] == 0
        assert pool.order[idx] == 0
        assert pool.target_idx[idx] == -1
        assert pool.target_kind[idx] == 0
        assert pool.comm[idx] == pytest.approx(1.0)
        assert pool.contact_age[idx] == pytest.approx(1e9)
        assert pool.formation_id[idx] == -1

    def test_spawn_ai_fields_args(self, pool: ArmorPool) -> None:
        """spawn 可显式传 AI 字段。"""
        idx = pool.spawn(10, 20, 0.0, side=1, role=4, formation_id=3)
        assert pool.side[idx] == 1
        assert pool.role[idx] == 4
        assert pool.formation_id[idx] == 3
        assert pool.target_idx[idx] == -1  # 其余保持默认

    def test_ai_fields_expand(self, pool: ArmorPool) -> None:
        """扩容时 AI 字段应同步扩容且保留旧数据。"""
        idx = pool.spawn(0, 0, 0, side=1, role=4, formation_id=3)
        pool._expand(factor=2.0)
        assert pool.side.shape[0] == MAX_ARMOR * 2
        assert pool.role.shape[0] == MAX_ARMOR * 2
        assert pool.order.shape[0] == MAX_ARMOR * 2
        assert pool.target_idx.shape[0] == MAX_ARMOR * 2
        assert pool.target_kind.shape[0] == MAX_ARMOR * 2
        assert pool.comm.shape[0] == MAX_ARMOR * 2
        assert pool.contact_age.shape[0] == MAX_ARMOR * 2
        assert pool.formation_id.shape[0] == MAX_ARMOR * 2
        # 旧数据保留
        assert pool.side[idx] == 1
        assert pool.role[idx] == 4
        assert pool.formation_id[idx] == 3
        # 新扩容区保持默认
        assert pool.formation_id[MAX_ARMOR] == -1
        assert pool.side[MAX_ARMOR] == 0


# ---------------------------------------------------------------------------
# ProjectilePool
# ---------------------------------------------------------------------------


class TestProjectilePool:
    """ProjectilePool 测试套件。"""

    @pytest.fixture
    def pool(self) -> ProjectilePool:
        return ProjectilePool()

    def test_spawn_basic(self, pool: ProjectilePool) -> None:
        """spawn 应设置位置、速度、伤害和类型。"""
        idx = pool.spawn(0, 0, 100, 800, 0, 0, radius=0.05, dmg=50, kind=ProjectileType.BULLET)
        assert pool.active[idx]
        assert pool.pos[idx, 2] == pytest.approx(100)
        assert pool.vel[idx, 0] == pytest.approx(800)
        assert pool.dmg[idx] == pytest.approx(50)
        assert pool.kind[idx] == int(ProjectileType.BULLET)

    def test_spawn_missile_type(self, pool: ProjectilePool) -> None:
        """导弹类型应正确标记。"""
        idx = pool.spawn(0, 0, 0, 300, 0, 0, kind=ProjectileType.MISSILE)
        assert pool.kind[idx] == int(ProjectileType.MISSILE)

    def test_spawn_owner_default(self, pool: ProjectilePool) -> None:
        """默认 owner 应为 -1（无主）。"""
        idx = pool.spawn(0, 0, 0, 100, 0, 0)
        assert pool.owner[idx] == -1

    def test_spawn_owner_set(self, pool: ProjectilePool) -> None:
        """owner 参数应正确存储。"""
        idx = pool.spawn(0, 0, 0, 100, 0, 0, owner=42)
        assert pool.owner[idx] == 42

    def test_destroy_clears_owner(self, pool: ProjectilePool) -> None:
        """destroy 应清除 owner。"""
        idx = pool.spawn(0, 0, 0, 100, 0, 0, owner=7)
        pool.destroy(idx)
        assert pool.owner[idx] == -1
        assert not pool.active[idx]

    def test_destroy_batch(self, pool: ProjectilePool) -> None:
        """批量销毁应正确。"""
        indices = [pool.spawn(float(i), 0, 0, 100, 0, 0) for i in range(10)]
        arr = np.array(indices[:5], dtype=np.int32)
        pool.destroy_batch(arr)
        assert pool.active_count == 5
        for i in arr:
            assert not pool.active[i]
            assert pool.owner[i] == -1

    def test_active_count_and_indices(self, pool: ProjectilePool) -> None:
        """active_count 和 active_indices 应一致。"""
        idx = pool.spawn(0, 0, 0, 100, 0, 0)
        assert pool.active_count == 1
        assert list(pool.active_indices()) == [idx]


# ---------------------------------------------------------------------------
# BoxPool
# ---------------------------------------------------------------------------


class TestBoxPool:
    """BoxPool 测试套件。"""

    @pytest.fixture
    def pool(self) -> BoxPool:
        return BoxPool()

    def test_spawn_basic(self, pool: BoxPool) -> None:
        """spawn 应设置父装甲、部位和半边长。"""
        idx = pool.spawn(
            parent_idx=5, part=PartType.TURRET,
            half_x=1.0, half_y=2.0, half_z=0.8,
        )
        assert pool.active[idx]
        assert pool.parent[idx] == 5
        assert pool.part[idx] == int(PartType.TURRET)
        assert pool.half[idx, 0] == pytest.approx(1.0)
        assert pool.half[idx, 1] == pytest.approx(2.0)

    def test_spawn_local_offset(self, pool: BoxPool) -> None:
        """局部偏移应存入 pos（sync 前为局部偏移）。"""
        idx = pool.spawn(
            parent_idx=0, part=PartType.HULL,
            half_x=1, half_y=3, half_z=1,
            local_x=0.5, local_y=-2.0, local_z=0.3,
        )
        # sync 前 pos 为局部偏移
        assert pool.pos[idx, 0] == pytest.approx(0.5)
        assert pool.pos[idx, 1] == pytest.approx(-2.0)

    def test_destroy_clears_parent(self, pool: BoxPool) -> None:
        """destroy 应清除 parent。"""
        idx = pool.spawn(parent_idx=3, part=PartType.ENGINE, half_x=1, half_y=1, half_z=1)
        pool.destroy(idx)
        assert not pool.active[idx]
        assert pool.parent[idx] == -1

    def test_sync_from_parent(self, pool: BoxPool) -> None:
        """sync_from_parent 应把局部偏移变换到世界坐标。"""
        armor = ArmorPool()
        aidx = armor.spawn(10, 20, 5, yaw=np.pi / 2)  # 朝向 +Y
        bidx = pool.spawn(
            parent_idx=aidx, part=PartType.TURRET,
            half_x=1, half_y=2, half_z=1,
            local_x=3, local_y=1, local_z=0.5,
        )
        pool.sync_from_parent(bidx, armor)
        # yaw=π/2: cos=0, sin=1 → (10 + 3*0 - 1*1, 20 + 3*1 + 1*0)
        assert pool.pos[bidx, 0] == pytest.approx(9.0)
        assert pool.pos[bidx, 1] == pytest.approx(23.0)
        assert pool.pos[bidx, 2] == pytest.approx(5.5)

    def test_sync_from_parent_dead_armor_deactivates(self, pool: BoxPool) -> None:
        """父装甲已销毁时 sync 应停用碰撞箱。"""
        armor = ArmorPool()
        aidx = armor.spawn(0, 0, 0)
        bidx = pool.spawn(parent_idx=aidx, part=PartType.HULL, half_x=1, half_y=1, half_z=1)
        armor.destroy(aidx)
        pool.sync_from_parent(bidx, armor)
        assert not pool.active[bidx]


# ---------------------------------------------------------------------------
# FormationAIPool（Phase 6c，docs/phase6-ai-design.md §1.2）
# ---------------------------------------------------------------------------


class TestFormationAIPool:
    """FormationAIPool 测试套件（编队 AI 决策主体池）。"""

    @pytest.fixture
    def pool(self) -> FormationAIPool:
        return FormationAIPool()

    def test_spawn_defaults(self, pool: FormationAIPool) -> None:
        """spawn 应登记编队并设置默认值。"""
        idx = pool.spawn()
        assert pool.active[idx]
        assert pool.side[idx] == 0
        assert pool.role[idx] == 0
        assert pool.intent[idx] == 0
        assert pool.order[idx] == 0
        assert pool.stance[idx] == 0
        assert pool.comm[idx] == pytest.approx(1.0)
        assert pool.member_count[idx] == 0
        assert pool.contact_set[idx] is None

    def test_spawn_args(self, pool: FormationAIPool) -> None:
        """spawn 可显式传阵营/角色/目标区。"""
        idx = pool.spawn(side=1, role=2, target_zone=(100.0, 200.0, 5.0))
        assert pool.side[idx] == 1
        assert pool.role[idx] == 2
        assert pool.target_zone[idx].tolist() == [100.0, 200.0, 5.0]

    def test_active_indices_and_count(self, pool: FormationAIPool) -> None:
        """active_count / active_indices 应正确。"""
        a = pool.spawn()
        b = pool.spawn()
        assert pool.active_count == 2
        assert set(pool.active_indices().tolist()) == {a, b}
        pool.destroy(b)
        assert pool.active_count == 1
        assert pool.active_indices().tolist() == [a]

    def test_destroy_clears_contact(self, pool: FormationAIPool) -> None:
        """destroy 应清空 contact_set 并归还索引。"""
        idx = pool.spawn()
        pool.contact_set[idx] = [{"idx": 1}]
        pool.destroy(idx)
        assert not pool.active[idx]
        assert pool.contact_set[idx] is None
        assert pool.spawn() == idx  # 索引复用

    def test_expand_preserves_data(self) -> None:
        """扩容应同步数组并保留旧数据。"""
        pool = FormationAIPool(capacity=2)
        idx = pool.spawn(side=1, role=4, target_zone=(1.0, 2.0, 3.0))
        pool.spawn()
        pool.spawn()  # 触发扩容 2→4
        assert pool.capacity == 4
        assert pool.active_count == 3
        assert pool.side[idx] == 1
        assert pool.role[idx] == 4
        assert pool.target_zone[idx].tolist() == [1.0, 2.0, 3.0]
        assert len(pool.contact_set) == 4

    def test_max_formations_constant(self) -> None:
        """默认容量应为 MAX_FORMATIONS。"""
        assert FormationAIPool().capacity == MAX_FORMATIONS
